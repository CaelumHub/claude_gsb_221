"""
repair.py — Spectral repair (频谱修复).

The user draws one or more rectangles on the spectrogram — each rectangle is a
time–frequency region — and the content inside every region is attenuated by an
adjustable amount (down to full erasure).  Everything outside the regions is
left untouched, the duration is preserved exactly, and no clicks are introduced.

Algorithm
---------
A classic STFT magnitude mask, but evaluated as a *difference* signal:

    out = istft(mask * stft(x))
        = istft(stft(x)) + istft((mask - 1) * stft(x))
        = x + istft(diff)

Because the WOLA overlap-add reconstruction (:func:`dsp.istft`) is linear and
the Hann window at ``hop = nfft/4`` satisfies COLA, ``istft(stft(x))`` is
exactly ``x``.  The diff frames are zero for every frame that no region
touches, so only frames intersecting a selection ever get transformed — a small
selection on a long file costs a handful of FFTs, and untouched audio passes
through bit-transparently.

Mask shape
----------
Each region produces a separable mask ``w_t(frame) * w_f(bin)`` built from the
selection rectangle with any-touch semantics (a frame/bin whose cell intersects
the rectangle is selected, so even a tiny box always affects at least one
frame and one bin).  The profile is flat (full depth) across the selected cells
and falls off with raised-cosine ramps at the edges — this is what keeps the
processed audio free of clicks without weakening narrow selections.  Overlapping
regions combine multiplicatively, so their attenuations compound instead of
interfering.

The mask only ever *reduces* bin magnitudes (gain <= 1) and phase is preserved,
so the overall balance and loudness of the file are maintained.
"""

from __future__ import annotations

import io
import math
import struct
from typing import Dict, List, Sequence, Tuple

from . import audio_io, dsp

# Smoothing of the mask edges (raised-cosine ramps).  The defaults give a
# ~2-frame time ramp (~23 ms at 44.1 kHz) and a ~2-bin frequency ramp (~43 Hz),
# which is comfortably click-free without visibly widening the selection.
DEFAULT_EDGE_MS = 15.0
DEFAULT_EDGE_HZ = 40.0

MAX_REGIONS = 64  # sanity limit for a single repair request


# --------------------------------------------------------------------------- #
# Region normalisation
# --------------------------------------------------------------------------- #

def _build_profile(lo: int, hi: int, ramp: int) -> Tuple[int, List[float]]:
    """Smoothed selection profile for cells [lo, hi].

    Returns ``(start, values)`` where ``values[i]`` is the mask weight of cell
    ``start + i``.  The profile is exactly 1 across the selected cells and,
    with ``ramp > 0``, extends ``ramp`` cells beyond both ends with a
    raised-cosine falloff, so the gain transitions smoothly to 1.0 (no clicks)
    while even a single-cell selection keeps its full depth.
    """
    if ramp <= 0:
        return lo, [1.0] * (hi - lo + 1)
    start = lo - ramp
    values = []
    for i in range(start, hi + ramp + 1):
        if lo <= i <= hi:
            values.append(1.0)
        elif i < lo:
            t = (i - start) / ramp
            values.append(0.5 - 0.5 * math.cos(math.pi * t))
        else:
            t = (hi + ramp - i) / ramp
            values.append(0.5 - 0.5 * math.cos(math.pi * t))
    # The first/last entries are exactly zero — drop them so they don't pull
    # extra frames into the transform set.
    while values and values[0] == 0.0:
        values.pop(0)
        start += 1
    while values and values[-1] == 0.0:
        values.pop()
    return start, values


def _span_to_indices(lo_val: float, hi_val: float, cell: float, count: int) -> Tuple[int, int]:
    """Map a [lo, hi] interval (seconds or Hz) to intersecting cell indices.

    Cells are centred at ``i * cell`` with half-width ``cell/2`` and tile the
    axis, so any interval — however small — intersects at least one cell.
    """
    i_lo = int(math.floor((lo_val - 0.5 * cell) / cell)) + 1
    i_hi = int(math.ceil((hi_val + 0.5 * cell) / cell)) - 1
    i_lo = max(0, min(i_lo, count - 1))
    i_hi = max(0, min(i_hi, count - 1))
    if i_lo > i_hi:  # degenerate (zero-width) selection -> the centre cell
        c = int(0.5 * (lo_val + hi_val) / cell)
        c = max(0, min(c, count - 1))
        i_lo = i_hi = c
    return i_lo, i_hi


class _Region:
    """A normalised selection rectangle with its smoothed mask profiles."""

    __slots__ = ("gain", "f_start", "wt", "k_start", "wf")

    def __init__(self, gain: float, f_start: int, wt: List[float],
                 k_start: int, wf: List[float]):
        self.gain = gain          # linear amplitude gain inside the region (0..1)
        self.f_start = f_start    # first frame index covered by ``wt``
        self.wt = wt              # time profile (len frames)
        self.k_start = k_start    # first bin index covered by ``wf``
        self.wf = wf              # frequency profile (len bins)

    def time_weight(self, frame: int) -> float:
        j = frame - self.f_start
        return self.wt[j] if 0 <= j < len(self.wt) else 0.0


def _prepare_regions(regions: Sequence[Dict], sr: float, duration: float,
                     n_frames: int, nbins: int, nfft: int, hop: int,
                     edge_ms: float, edge_hz: float) -> List[_Region]:
    """Validate + normalise the raw region dicts into mask profiles."""
    cell_t = hop / sr
    cell_f = sr / nfft
    ramp_t = 0 if edge_ms <= 0 else max(2, int(round(edge_ms / 1000.0 / cell_t)))
    ramp_f = 0 if edge_hz <= 0 else max(2, int(round(edge_hz / cell_f)))
    nyquist = sr / 2.0

    out: List[_Region] = []
    for raw in regions[:MAX_REGIONS]:
        t0 = dsp.clamp(float(raw.get("t0", 0.0)), 0.0, duration)
        t1 = dsp.clamp(float(raw.get("t1", 0.0)), 0.0, duration)
        f0 = dsp.clamp(float(raw.get("f0", 0.0)), 0.0, nyquist)
        f1 = dsp.clamp(float(raw.get("f1", 0.0)), 0.0, nyquist)
        if t1 < t0:
            t0, t1 = t1, t0
        if f1 < f0:
            f0, f1 = f1, f0
        gain_db = min(0.0, float(raw.get("gain_db", -24.0)))
        gain = 0.0 if gain_db <= -100.0 else 10.0 ** (gain_db / 20.0)
        if gain >= 1.0:
            continue  # no attenuation requested -> nothing to do

        i_lo, i_hi = _span_to_indices(t0, t1, cell_t, n_frames)
        k_lo, k_hi = _span_to_indices(f0, f1, cell_f, nbins)

        f_start, wt = _build_profile(i_lo, i_hi, ramp_t)
        k_start, wf = _build_profile(k_lo, k_hi, ramp_f)
        # Clip profiles that extend past the signal boundaries.
        if f_start < 0:
            wt = wt[-f_start:]
            f_start = 0
        if f_start + len(wt) > n_frames:
            wt = wt[:n_frames - f_start]
        if k_start < 0:
            wf = wf[-k_start:]
            k_start = 0
        if k_start + len(wf) > nbins:
            wf = wf[:nbins - k_start]
        if not wt or not wf:
            continue
        out.append(_Region(gain, f_start, wt, k_start, wf))
    return out


# --------------------------------------------------------------------------- #
# Core: masked-STFT repair of one channel
# --------------------------------------------------------------------------- #

def _repair_channel(samples: Sequence[float], sr: float,
                    regions: List[_Region], nfft: int, hop: int) -> List[float]:
    """Return ``samples`` with every region attenuated; length is preserved."""
    n = len(samples)
    out = list(samples)
    if not regions or n == 0:
        return out

    pad = nfft // 2
    n_frames = n // hop + 1
    win = dsp.hann(nfft)
    w2 = [v * v for v in win]
    overlap = nfft // hop  # frames covering any interior position

    # Frames touched by at least one region, grouped into consecutive clusters
    # so the OLA buffers stay proportional to the processed span.
    touched = set()
    for reg in regions:
        for i in range(reg.f_start, reg.f_start + len(reg.wt)):
            touched.add(i)
    if not touched:
        return out
    ordered = sorted(touched)
    clusters: List[Tuple[int, int]] = []
    c_start = c_prev = ordered[0]
    for i in ordered[1:]:
        if i > c_prev + 1:
            clusters.append((c_start, c_prev))
            c_start = i
        c_prev = i
    clusters.append((c_start, c_prev))

    for a, b in clusters:
        pos0 = a * hop              # cluster start in padded-sample coordinates
        pos1 = b * hop + nfft       # cluster end (exclusive)
        length = pos1 - pos0
        diff = [0.0] * length
        norm = [0.0] * length

        # WOLA normalisation: every frame of the *full* grid whose window
        # covers this span contributes w^2 — including unmodified neighbours
        # (up to ``overlap - 1`` frames beyond the touched ones on each side).
        f_lo = max(0, a - overlap + 1)
        f_hi = min(n_frames - 1, b + overlap - 1)
        for f in range(f_lo, f_hi + 1):
            base = f * hop - pos0
            j0 = max(0, -base)
            j1 = min(nfft, length - base)
            for j in range(j0, j1):
                norm[base + j] += w2[j]

        for i in range(a, b + 1):
            active = [reg for reg in regions
                      if reg.f_start <= i < reg.f_start + len(reg.wt)]
            if not active:
                continue

            # Combined multiplicative mask over the union of affected bins.
            k_lo = min(reg.k_start for reg in active)
            k_hi = max(reg.k_start + len(reg.wf) - 1 for reg in active)
            mask = [1.0] * (k_hi - k_lo + 1)
            for reg in active:
                wt = reg.time_weight(i)
                if wt <= 0.0:
                    continue
                depth = (1.0 - reg.gain) * wt
                base = reg.k_start - k_lo
                for j, wf in enumerate(reg.wf):
                    mask[base + j] *= 1.0 - depth * wf

            # Windowed frame (centre-padded coordinates).
            start = i * hop - pad
            lo = max(0, start)
            hi = min(n, start + nfft)
            seg = ([0.0] * (lo - start)) + list(samples[lo:hi]) + \
                  ([0.0] * (start + nfft - hi))
            spec = dsp.fft([seg[j] * win[j] for j in range(nfft)])

            # Spectral difference, kept conjugate-symmetric -> real output.
            diffspec = [0.0j] * nfft
            for k in range(k_lo, k_hi + 1):
                m = mask[k - k_lo]
                if m >= 1.0:
                    continue
                d = spec[k] * (m - 1.0)
                diffspec[k] = d
                if 0 < k < nfft - k:
                    diffspec[nfft - k] = spec[nfft - k] * (m - 1.0)
            y = dsp.ifft(diffspec)

            base = i * hop - pos0
            for j in range(nfft):
                diff[base + j] += y[j].real * win[j]

        # Fold the normalised difference back into the signal.
        for p in range(length):
            nm = norm[p]
            if nm <= 1e-9:
                continue
            d = diff[p]
            if d == 0.0:
                continue
            idx = pos0 + p - pad
            if 0 <= idx < n:
                out[idx] += d / nm
    return out


# --------------------------------------------------------------------------- #
# Whole-file entry points
# --------------------------------------------------------------------------- #

def _auto_nfft(sr: float) -> int:
    """Pick an STFT size of ~46 ms, clamped to [256, 4096] and a power of two."""
    target = max(256, min(4096, int(sr * 0.046)))
    return dsp.next_pow2(target)


def _read_all(path: str) -> Tuple[List[List[float]], int]:
    """Read the whole file as de-interleaved float channels."""
    channels: List[List[float]] = []
    with audio_io.WavReader(path) as r:
        channels = [[] for _ in range(r.channels)]
        for chunk in r.iter_chunks():
            for c, ch in enumerate(chunk):
                channels[c].extend(ch)
        sr = r.sr
    return channels, sr


def spectral_repair(src_path: str, dst_path: str, regions: Sequence[Dict],
                    nfft: int = 0, edge_ms: float = DEFAULT_EDGE_MS,
                    edge_hz: float = DEFAULT_EDGE_HZ) -> Dict:
    """Attenuate/erase every region of ``src_path`` and write ``dst_path``.

    ``regions`` is a list of ``{"t0", "t1", "f0", "f1", "gain_db"}`` dicts
    (seconds / Hz / dB, gain_db <= 0; <= -100 dB means full erasure).
    The output has exactly the same duration, sample rate and channel count as
    the input; audio outside the regions is passed through unchanged.
    """
    if not regions:
        raise ValueError("no repair regions given")

    channels, sr = _read_all(src_path)
    n = len(channels[0]) if channels else 0
    if n == 0:
        raise ValueError("empty audio file")
    duration = n / sr

    if nfft <= 0:
        nfft = _auto_nfft(sr)
    hop = nfft // 4
    n_frames = n // hop + 1
    nbins = nfft // 2 + 1

    prepared = _prepare_regions(regions, sr, duration, n_frames, nbins,
                                nfft, hop, edge_ms, edge_hz)
    if not prepared:
        # Every region was a no-op (e.g. gain 0 dB): copy through unchanged.
        audio_io.save(dst_path, audio_io.AudioData(channels, sr))
        return {"frames": n, "sr": sr, "channels": len(channels),
                "regions_applied": 0, "nfft": nfft, "hop": hop}

    out_channels = []
    for ch in channels:
        repaired = _repair_channel(ch, sr, prepared, nfft, hop)
        # Safety limiter: attenuation cannot raise the level, but if a
        # pathological reconstruction overshoots both 0 dBFS and the input
        # peak, pull it back rather than let the 16-bit encoder hard-clip.
        peak_in = max((abs(v) for v in ch), default=0.0)
        peak = max((abs(v) for v in repaired), default=0.0)
        if peak > 1.0 and peak > peak_in:
            g = min(1.0, peak_in) / peak
            repaired = [v * g for v in repaired]
        out_channels.append(repaired)

    audio_io.save(dst_path, audio_io.AudioData(out_channels, sr))
    return {"frames": n, "sr": sr, "channels": len(channels),
            "regions_applied": len(prepared), "nfft": nfft, "hop": hop}


def repair_preview(src_path: str, regions: Sequence[Dict],
                   gain_db: float = -24.0, context_s: float = 1.0,
                   max_seconds: float = 15.0) -> bytes:
    """Render the repaired span (plus ``context_s`` of context) as WAV bytes.

    Only the excerpt covering the regions is processed and returned, so the
    preview is fast even for long files.
    """
    if not regions:
        raise ValueError("no repair regions given")

    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        total = r.nframes
        t0 = min(float(reg.get("t0", 0.0)) for reg in regions)
        t1 = max(float(reg.get("t1", 0.0)) for reg in regions)
        start = max(0.0, t0 - context_s)
        end = min(total / sr, t1 + context_s)
        if end - start > max_seconds:
            end = start + max_seconds
        start_f = int(start * sr)
        n_f = max(1, int((end - start) * sr))
        excerpt = r.read_excerpt(start_f, n_f)

    n = excerpt.frames
    duration = n / sr if sr else 0.0
    # Shift region times into excerpt coordinates and drop non-overlapping ones.
    local = []
    for reg in regions:
        lt0 = float(reg.get("t0", 0.0)) - start
        lt1 = float(reg.get("t1", 0.0)) - start
        if lt1 <= 0.0 or lt0 >= duration:
            continue
        local.append({"t0": lt0, "t1": lt1,
                      "f0": float(reg.get("f0", 0.0)),
                      "f1": float(reg.get("f1", 0.0)),
                      "gain_db": float(reg.get("gain_db", gain_db))})
    if not local:
        raise ValueError("regions do not overlap the preview window")

    nfft = _auto_nfft(sr)
    hop = nfft // 4
    prepared = _prepare_regions(local, sr, duration, n // hop + 1,
                                nfft // 2 + 1, nfft, hop,
                                DEFAULT_EDGE_MS, DEFAULT_EDGE_HZ)

    out_channels = []
    for ch in excerpt.samples:
        repaired = _repair_channel(ch, sr, prepared, nfft, hop) if prepared else list(ch)
        out_channels.append(dsp.fade_in_out(repaired, 0.01, sr))

    return _wav_bytes(out_channels, sr)


def _wav_bytes(channels: Sequence[Sequence[float]], sr: int) -> bytes:
    """Encode de-interleaved float channels as a 16-bit PCM WAV byte string."""
    n_ch = len(channels)
    n_frames = min((len(c) for c in channels), default=0)
    data = bytearray()
    for i in range(n_frames):
        for c in channels:
            v = int(round(max(-1.0, min(1.0, c[i])) * 32767))
            data += struct.pack("<h", v)
    byte_rate = sr * n_ch * 2
    block_align = n_ch * 2
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_ch, sr, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(bytes(data))
    return buf.getvalue()
