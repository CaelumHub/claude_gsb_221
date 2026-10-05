"""
spectral.py — Spectral repair ("频谱修复"): erase or attenuate unwanted
time-frequency regions on a spectrogram.

The user draws one or more rectangles on the spectrogram — each rectangle
selects a time span and a frequency band (e.g. a whistle, a hum, a narrow-band
noise burst).  Inside every selected rectangle the magnitude spectrum is
multiplied by a constant gain (1.0 = untouched, 0.0 = fully erased, anything
in between = attenuation); outside the rectangles the signal passes through
unchanged.  The audio is re-synthesised from the modified STFT with the
original phase via weighted overlap-add ISTFT.

Artefact-free design
--------------------
* **Feathered rectangle edges.**  Every edge of every rectangle gets a raised-
  cosine transition (``feather_t`` seconds in time, ``feather_f`` Hz in
  frequency).  Hard mask edges cause pre-echo / ringing / clicks; the smooth
  fade makes the boundary inaudible.  A rectangle that touches DC, Nyquist,
  the file start or the file end simply clamps the fade at the boundary, so
  full-band and "spans the whole spectrum" selections work naturally.
* **Tiny selections.**  The feather width is automatically shrunk to half the
  rectangle extent, so a one-frame / one-bin selection still attenuates its
  centre correctly instead of being washed out.
* **Multiple / overlapping rectangles.**  Their gain fields are multiplied
  (minimum-gain for attenuation), which is smooth even when overlapped.
* **Outside is bit-for-bit preserved.**  Only the time segments covered by a
  rectangle (plus one FFT window of margin on each side) go through the STFT
  round-trip; the rest is copied sample-by-sample.  Segments are joined with a
  short equal-power crossfade, so no click is introduced at the seams and the
  output duration / sample count is identical to the input.
* **No new distortion.**  Attenuation only lowers magnitudes; after the WOLA
  reconstruction a safety guard scales the segment down in the unlikely event
  that the round-trip overshoots the original segment peak.

The implementation is pure Python on top of :mod:`backend.dsp` (radix-2 FFT +
overlap-add), consistent with the rest of the project.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import audio_io, dsp

# --------------------------------------------------------------------------- #
# Rectangle gain fields
# --------------------------------------------------------------------------- #


def _edge_weight(x: float, lo: float, hi: float, feather: float) -> float:
    """Smooth membership weight (0 outside, 1 deep inside) of scalar ``x`` in
    the interval [lo, hi] with raised-cosine edges of width ``feather``.
    """
    if x < lo or x > hi:
        return 0.0
    # Shrink the feather so it never exceeds half the interval extent; this
    # keeps tiny selections effective (their centre still reaches weight 1).
    feather = min(feather, max(0.0, (hi - lo) * 0.5))
    if feather <= 0.0:
        return 1.0
    if x < lo + feather:
        u = (x - lo) / feather
        return 0.5 - 0.5 * math.cos(math.pi * u)
    if x > hi - feather:
        u = (hi - x) / feather
        return 0.5 - 0.5 * math.cos(math.pi * u)
    return 1.0


def _normalise_regions(regions: Sequence[Dict]) -> List[Tuple[float, float, float, float, float]]:
    """Validate rectangles and return sorted (t0, t1, f0, f1, gain) tuples."""
    out: List[Tuple[float, float, float, float, float]] = []
    for reg in regions:
        t0 = float(reg.get("start", reg.get("t0", 0.0)))
        t1 = float(reg.get("end", reg.get("t1", 0.0)))
        f0 = float(reg.get("f_low", reg.get("f0", 0.0)))
        f1 = float(reg.get("f_high", reg.get("f1", 0.0)))
        t0, t1 = min(t0, t1), max(t0, t1)
        f0, f1 = min(f0, f1), max(f0, f1)
        if t1 <= t0 or f1 <= f0:
            continue
        # Attenuation strength: reduction_db > 0, or gain 0..1, or erase flag.
        if reg.get("erase"):
            gain = 0.0
        elif "gain" in reg:
            gain = min(1.0, max(0.0, float(reg["gain"])))
        else:
            db = float(reg.get("reduction_db", reg.get("attenuation_db", 96.0)))
            gain = 10.0 ** (-max(0.0, db) / 20.0)
        out.append((t0, t1, max(0.0, f0), f1, gain))
    out.sort(key=lambda r: r[0])
    return out


def _build_gain_table(
    n_frames: int,
    n_bins: int,
    nfft: int,
    hop: int,
    sr: float,
    regions: Sequence[Tuple[float, float, float, float, float]],
    feather_t: float,
    feather_f: float,
    excerpt_start: int = 0,
) -> List[List[float]]:
    """Per-frame, per-bin multiplicative gain (n_frames x n_bins).

    ``excerpt_start`` is the sample offset of the first frame's segment within
    the full file (used to map frame centres onto global selection times).
    Frames fully outside every rectangle keep an all-ones row.
    """
    gains: List[List[float]] = [[1.0] * n_bins for _ in range(n_frames)]
    if not regions:
        return gains

    nyquist = sr / 2.0
    dt_frame = hop / sr
    df_bin = sr / nfft
    for t_idx in range(n_frames):
        # With dsp.stft(center=True) frame t is centred on this global sample.
        centre_t = (excerpt_start - nfft // 2 + t_idx * hop + nfft // 2) / sr
        active = [r for r in regions if r[0] - 0.5 * dt_frame <= centre_t <= r[1] + 0.5 * dt_frame]
        if not active:
            continue
        row = gains[t_idx]
        for k in range(n_bins):
            f = k * sr / nfft
            g = 1.0
            for (t0, t1, f0, f1, gain) in active:
                # Quantise the rectangle outward by half an analysis cell so a
                # selection narrower than one hop / one bin still addresses the
                # nearest frame / bin (the physical resolution limit).
                qt0, qt1 = t0 - 0.5 * dt_frame, t1 + 0.5 * dt_frame
                qf0, qf1 = f0 - 0.5 * df_bin, min(f1 + 0.5 * df_bin, nyquist)
                # Each rectangle gets an effective feather no wider than half
                # its own (quantised) extent and no narrower than the analysis
                # resolution, so tiny / ultra-narrow selections still attenuate
                # their centre instead of being washed out by a broad fade.
                ft_eff = min(feather_t, (qt1 - qt0) * 0.5, 2.0 * dt_frame)
                ff_eff = min(feather_f, (qf1 - qf0) * 0.5, 2.0 * df_bin)
                wt = _edge_weight(centre_t, qt0, qt1, ft_eff)
                wf = _edge_weight(f, qf0, qf1, ff_eff)
                w = wt * wf
                if w > 0.0:
                    g *= 1.0 - (1.0 - gain) * w
            row[k] = g
    return gains


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #


def _choose_stft_params(
    rects: Sequence[Tuple[float, float, float, float, float]],
    sr: float,
    nfft: Optional[int] = None,
    hop: Optional[int] = None,
) -> Tuple[int, int]:
    """Pick an FFT size/hop that lets every rectangle be addressed.

    Time and frequency resolution trade off (nfft sets both: hop = nfft/4
    gives ~4 frame centres per window, bin spacing is sr/nfft).  We anchor at
    nfft = 2048 (a good general-purpose choice) and then:

      * shrink it (down to 128) when the shortest selection is too brief to
        hold two frame centres at the current size;
      * grow it (up to 8192) when the narrowest band contains fewer than two
        bins at the current size.

    Selections that are simultaneously extremely short and extremely narrow
    live below the uncertainty principle and simply take the smaller (time
    localisation) size.  Explicit ``nfft`` / ``hop`` always win.
    """
    if nfft:
        n = int(nfft)
        return n, int(hop) if hop else n // 4

    min_t = min((r[1] - r[0] for r in rects), default=1.0)
    min_f = min((r[3] - r[2] for r in rects), default=sr / 4)
    n = 2048
    # Short selection: shrink while the window spans more than ~1.5 times the
    # selection (then no frame centre reaches the deep-interior weight of 1).
    while n > 128 and n / sr > 1.5 * min_t:
        n >>= 1
    # Narrow band: grow while fewer than ~2 bins span it — only as long as the
    # time-localisation budget allows.
    while n < 8192 and (n * min_f / sr) < 2.0 and (n << 1) / sr <= 1.5 * min_t:
        n <<= 1
    return dsp.next_pow2(n), dsp.next_pow2(n) // 4


def _merge_intervals(intervals: Sequence[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    """Merge integer intervals whose gap is smaller than ``gap`` samples."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def repair_samples(
    channels: Sequence[List[float]],
    sr: float,
    regions: Sequence[Dict],
    feather_t: float = 0.01,
    feather_f: float = 40.0,
    nfft: int = 2048,
    hop: int = 512,
    excerpt_start: int = 0,
) -> List[List[float]]:
    """Apply spectral repair to an in-memory excerpt.

    ``channels`` holds one float list per channel; ``excerpt_start`` is the
    excerpt's offset within the full file in samples.  Returns new channel
    lists of exactly the same length.  Rectangles are given in global time /
    frequency coordinates.
    """
    rects = _normalise_regions(regions)
    n = min((len(c) for c in channels), default=0)
    if n == 0:
        return [[] for _ in channels]
    if not rects:
        return [list(c[:n]) for c in channels]

    nfft = max(64, int(nfft))
    hop = max(1, min(int(hop), nfft // 2))
    feather_t = max(0.0, float(feather_t))
    feather_f = max(0.0, float(feather_f))

    out_ch: List[List[float]] = []
    for ch in channels:
        sig = list(ch[:n])
        frames = dsp.stft(sig, nfft, hop, "hann", center=True)
        n_bins = nfft // 2 + 1
        gains = _build_gain_table(
            len(frames), n_bins, nfft, hop, sr, rects,
            feather_t, feather_f, excerpt_start,
        )

        mod_frames: List[List[complex]] = []
        for t_idx, fr in enumerate(frames):
            row = gains[t_idx]
            if not row or max(abs(g - 1.0) for g in row) < 1e-12:
                mod_frames.append(fr)  # untouched frame
                continue
            mf = list(fr)
            for k in range(n_bins):
                g = row[k]
                mf[k] = fr[k] * g
                if 0 < k < nfft // 2:  # keep the modified spectrum conjugate-symmetric
                    mf[nfft - k] = fr[nfft - k] * g
            mod_frames.append(mf)

        rec = dsp.istft(mod_frames, nfft, hop, "hann", length=len(sig) + nfft)
        rec = rec[nfft // 2:nfft // 2 + n]

        # Safety guard: attenuation must never make the segment louder.
        orig_peak = max((abs(v) for v in sig), default=0.0)
        new_peak = max((abs(v) for v in rec), default=0.0)
        if orig_peak > 1e-12 and new_peak > orig_peak:
            scale = orig_peak / new_peak
            rec = [v * scale for v in rec]
        out_ch.append(rec)
    return out_ch


def repair_file(
    src_path: str,
    dst_path: str,
    regions: Sequence[Dict],
    feather_t: float = 0.01,
    feather_f: float = 40.0,
    nfft: Optional[int] = None,
    hop: Optional[int] = None,
    reduction_db: Optional[float] = None,
) -> Dict:
    """Repair a WAV file, writing a same-length, same-rate WAV to ``dst_path``.

    Only the sample ranges touched by the selections (plus one FFT window of
    margin each side) are transformed; everything else is copied verbatim.
    ``reduction_db`` overrides every rectangle's per-region strength (the UI
    uses it for the single global "衰减强度" slider).  ``nfft``/``hop``
    default to an adaptive resolution fine enough for the smallest selection.
    """
    if reduction_db is not None:
        regions = [dict(r, reduction_db=float(reduction_db)) for r in regions]
    rects = _normalise_regions(regions)

    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        ch_count = r.channels
        total = r.nframes

    if not rects:
        # Nothing to do: copy the file through unchanged.
        with audio_io.WavReader(src_path) as rin, \
             audio_io.WavWriter(dst_path, sr, ch_count, 2) as wout:
            for chunk in rin.iter_chunks():
                wout.write_chunk(chunk)
        return {"frames": total, "regions": 0, "sr": sr, "channels": ch_count}

    nfft, hop = _choose_stft_params(rects, sr, nfft, hop)
    margin = nfft
    intervals: List[Tuple[int, int]] = []
    for (t0, t1, f0, f1, gain) in rects:
        lo = max(0, int(t0 * sr) - margin)
        hi = min(total, int(math.ceil(t1 * sr)) + margin)
        if hi > lo:
            intervals.append((lo, hi))
    # Join intervals separated by less than a window so crossfades never overlap.
    segments = _merge_intervals(intervals, gap=2 * nfft)

    # Only positive-frequency content can be attenuated; clamp frequency edges
    # to Nyquist (rectangles "spanning the whole band" are simply passed as-is).
    xfade = max(8, min(nfft // 2, 256))

    with audio_io.WavReader(src_path) as rin, \
         audio_io.WavWriter(dst_path, sr, ch_count, 2) as wout:

        def _copy(pos: int, end: int) -> None:
            rin._w.setpos(pos)
            remaining = end - pos
            while remaining > 0:
                chunk = rin.read_chunk(min(1 << 16, remaining))
                if chunk is None:
                    break
                wout.write_chunk(chunk)
                remaining -= len(chunk[0])

        pos = 0
        for seg_lo, seg_hi in segments:
            if seg_lo > pos:
                _copy(pos, seg_lo)
            # Read the excerpt and run the STFT repair on it.
            rin._w.setpos(seg_lo)
            excerpt: List[List[float]] = [[] for _ in range(ch_count)]
            remaining = seg_hi - seg_lo
            while remaining > 0:
                chunk = rin.read_chunk(min(1 << 16, remaining))
                if chunk is None:
                    break
                for c in range(ch_count):
                    excerpt[c].extend(chunk[c])
                remaining -= len(chunk[0])
            repaired = repair_samples(
                excerpt, sr, regions, feather_t, feather_f,
                nfft, hop, excerpt_start=seg_lo,
            )

            # Equal-power crossfade at both seams against the verbatim samples.
            # The segment was extracted with a full FFT window of margin, so the
            # fade region is reconstruction-exact and the join is click-free.
            nseg = seg_hi - seg_lo
            rin._w.setpos(seg_lo)
            orig_chunk = rin.read_chunk(nseg)
            if orig_chunk is None:
                orig_chunk = [[0.0] * nseg for _ in range(ch_count)]
            for c in range(ch_count):
                seg = repaired[c]
                orig = orig_chunk[c]
                m = min(xfade, nseg // 2)
                for i in range(m):
                    u = i / max(1, m - 1)
                    w_new = 0.5 - 0.5 * math.cos(0.5 * math.pi * u)  # 0 -> 1
                    w_old = math.sqrt(max(0.0, 1.0 - w_new * w_new))
                    # Start seam: ramp the processed segment in.
                    seg[i] = orig[i] * w_old + seg[i] * w_new
                    # End seam (mirrored): ramp it back out toward the edge.
                    j = nseg - 1 - i
                    seg[j] = orig[j] * w_old + seg[j] * w_new
                repaired[c] = seg

            wout.write_chunk(repaired)
            pos = seg_hi

        if pos < total:
            _copy(pos, total)

    return {
        "frames": total,
        "regions": len(rects),
        "segments": len(segments),
        "sr": sr,
        "channels": ch_count,
        "nfft": nfft,
        "hop": hop,
    }


# --------------------------------------------------------------------------- #
# Preview (short in-memory WAV)
# --------------------------------------------------------------------------- #


def render_preview_wav(
    src_path: str,
    regions: Sequence[Dict],
    start_s: float,
    duration_s: float,
    feather_t: float = 0.01,
    feather_f: float = 40.0,
    nfft: Optional[int] = None,
    hop: Optional[int] = None,
    reduction_db: Optional[float] = None,
) -> bytes:
    """Render an excerpt through spectral repair and return WAV bytes."""
    import io
    import struct

    if reduction_db is not None:
        regions = [dict(r, reduction_db=float(reduction_db)) for r in regions]

    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        total = r.nframes
        nfft, hop = _choose_stft_params(_normalise_regions(regions), sr, nfft, hop)
        start = max(0, min(total - 1, int(start_s * sr)))
        n = max(0, min(int(duration_s * sr), total - start))
        r._w.setpos(start)
        chunk = r.read_chunk(n) if n > 0 else None
        if chunk is None:
            chunk = [[0.0] for _ in range(r.channels)]

    repaired = repair_samples(
        chunk, sr, regions, feather_t, feather_f,
        nfft, hop, excerpt_start=start,
    )

    buf = io.BytesIO()
    nframes = min(len(c) for c in repaired) if repaired else 0
    channels = len(repaired)
    data = bytearray()
    for i in range(nframes):
        for c in repaired:
            v = int(round(max(-1.0, min(1.0, c[i])) * 32767))
            data += struct.pack("<h", v)
    byte_rate = sr * channels * 2
    block_align = channels * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(bytes(data))
    return buf.getvalue()
