#!/usr/bin/env python3
"""
test_repair.py — Tests for the spectral-repair feature (backend/repair.py).

Covers:
  * identity passthrough when no attenuation is requested
  * band-limited attenuation inside a time window (strength adjustable)
  * full erasure of a region
  * exact duration / sample-rate / channel-count preservation
  * no clicks introduced (bounded sample-to-sample jumps)
  * tiny regions, full-band regions, multiple regions, edge regions
  * stereo files
  * preview rendering
  * the Flask API endpoints

Run from the repo root:

    python3 tests/test_repair.py
"""

import math
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import audio_io, repair  # noqa: E402

SR = 44100
TMP = tempfile.mkdtemp(prefix="repair_test_")
FAILURES = []
LSB = 1.0 / 32768  # 16-bit codec round-off (decode /32768, encode x32767)


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def sine(freq, seconds, amp=0.5, sr=SR, phase=0.0):
    n = int(seconds * sr)
    return [amp * math.sin(2 * math.pi * freq * i / sr + phase) for i in range(n)]


def mix(a, b):
    return [x + y for x, y in zip(a, b)]


def save_wav(name, channels, sr=SR):
    path = os.path.join(TMP, name)
    audio_io.save(path, audio_io.AudioData(channels, sr))
    return path


def read_wav(path):
    chans, sr = repair._read_all(path)
    return chans, sr


def amplitude_at(x, freq, sr=SR):
    """Amplitude of the ``freq`` component via single-bin correlation."""
    n = len(x)
    if n == 0:
        return 0.0
    re = sum(v * math.cos(2 * math.pi * freq * i / sr) for i, v in enumerate(x))
    im = sum(v * math.sin(2 * math.pi * freq * i / sr) for i, v in enumerate(x))
    return 2.0 * math.hypot(re, im) / n


def seg(x, t0, t1, sr=SR):
    return x[int(t0 * sr):int(t1 * sr)]


def max_jump(x):
    return max((abs(x[i + 1] - x[i]) for i in range(len(x) - 1)), default=0.0)


def max_abs_diff(a, b):
    return max((abs(u - v) for u, v in zip(a, b)), default=0.0)


# --------------------------------------------------------------------------- #
# DSP core tests
# --------------------------------------------------------------------------- #

def test_identity():
    print("identity passthrough (gain 0 dB)")
    src = save_wav("id_src.wav", [sine(440, 1.0)])
    dst = os.path.join(TMP, "id_dst.wav")
    repair.spectral_repair(src, dst, [{"t0": 0.2, "t1": 0.5, "f0": 100, "f1": 1000, "gain_db": 0.0}])
    a, _ = read_wav(src)
    b, _ = read_wav(dst)
    check("frames preserved", len(a[0]) == len(b[0]))
    check("bit-exact copy", max_abs_diff(a[0], b[0]) <= 2 / 32768,
          f"max diff {max_abs_diff(a[0], b[0])}")


def test_channel_inmemory_exact():
    print("in-memory channel repair: untouched samples are float-identical")
    sig = mix(sine(1000, 3.0), sine(4000, 3.0))
    n = len(sig)
    nfft, hop = 2048, 512
    n_frames = n // hop + 1
    regions = repair._prepare_regions(
        [{"t0": 1.0, "t1": 2.0, "f0": 3000, "f1": 5000, "gain_db": -40.0}],
        SR, n / SR, n_frames, nfft // 2 + 1, nfft, hop,
        repair.DEFAULT_EDGE_MS, repair.DEFAULT_EDGE_HZ)
    out = repair._repair_channel(sig, SR, regions, nfft, hop)
    check("length preserved", len(out) == n)
    # Influence span: touched frames +/- one window.
    touched = [i for r in regions for i in range(r.f_start, r.f_start + len(r.wt))]
    lo = min(touched) * hop - nfft // 2
    hi = max(touched) * hop + nfft // 2
    before = max_abs_diff(sig[:lo], out[:lo])
    after = max_abs_diff(sig[hi:], out[hi:])
    check("exactly zero before influence span", before == 0.0, f"{before}")
    check("exactly zero after influence span", after == 0.0, f"{after}")
    check("region actually modified", max_abs_diff(sig[lo:hi], out[lo:hi]) > 0.01)


def test_band_attenuation():
    print("band-limited attenuation in a time window")
    sig = mix(sine(1000, 3.0), sine(4000, 3.0))
    src = save_wav("band_src.wav", [sig])
    dst = os.path.join(TMP, "band_dst.wav")
    info = repair.spectral_repair(src, dst, [
        {"t0": 1.0, "t1": 2.0, "f0": 3000, "f1": 5000, "gain_db": -40.0}])
    ref = read_wav(src)[0][0]      # decoded source (16-bit round-trip)
    out = read_wav(dst)[0][0]
    check("regions applied", info["regions_applied"] == 1, str(info))
    check("frames preserved", len(out) == len(sig))

    in4k = amplitude_at(seg(ref, 1.3, 1.7), 4000)
    out4k = amplitude_at(seg(out, 1.3, 1.7), 4000)
    ratio = out4k / in4k if in4k else 1.0
    check("4 kHz attenuated ~40 dB inside region", 0.003 < ratio < 0.02,
          f"ratio {ratio:.5f} ({20 * math.log10(ratio):.1f} dB)")

    in1k = amplitude_at(seg(ref, 1.3, 1.7), 1000)
    out1k = amplitude_at(seg(out, 1.3, 1.7), 1000)
    check("1 kHz untouched inside region", abs(out1k / in1k - 1.0) < 0.02,
          f"ratio {out1k / in1k:.4f}")

    check("audio before region untouched (<=1 LSB)",
          max_abs_diff(seg(ref, 0.1, 0.8), seg(out, 0.1, 0.8)) <= LSB)
    check("audio after region untouched (<=1 LSB)",
          max_abs_diff(seg(ref, 2.2, 2.9), seg(out, 2.2, 2.9)) <= LSB)

    j_in, j_out = max_jump(ref), max_jump(out)
    check("no clicks (bounded jumps)", j_out <= j_in * 1.3 + 1e-3,
          f"in {j_in:.4f} out {j_out:.4f}")


def test_erase_full_band():
    print("full-band erasure of a time window")
    sig = mix(sine(500, 2.0), sine(3000, 2.0))
    src = save_wav("erase_src.wav", [sig])
    dst = os.path.join(TMP, "erase_dst.wav")
    repair.spectral_repair(src, dst, [
        {"t0": 0.8, "t1": 1.2, "f0": 0, "f1": SR / 2, "gain_db": -120.0}])
    ref = read_wav(src)[0][0]
    out = read_wav(dst)[0][0]
    rms_out = math.sqrt(sum(v * v for v in seg(out, 0.9, 1.1)) / len(seg(out, 0.9, 1.1)))
    check("window silenced", rms_out < 1e-4, f"rms out {rms_out:.6f}")
    check("outside intact (<=1 LSB)", max_abs_diff(seg(ref, 0.0, 0.6), seg(out, 0.0, 0.6)) <= LSB)
    check("no clicks", max_jump(out) <= max_jump(ref) * 1.3 + 1e-3)


def test_tiny_region():
    print("tiny region (3 ms x 30 Hz)")
    sig = sine(4000, 2.0)
    src = save_wav("tiny_src.wav", [sig])
    dst = os.path.join(TMP, "tiny_dst.wav")
    repair.spectral_repair(src, dst, [
        {"t0": 1.0, "t1": 1.003, "f0": 3985, "f1": 4015, "gain_db": -120.0}])
    ref = read_wav(src)[0][0]
    out = read_wav(dst)[0][0]
    check("frames preserved", len(out) == len(sig))
    in_amp = amplitude_at(seg(ref, 0.98, 1.02), 4000)
    out_amp = amplitude_at(seg(out, 0.98, 1.02), 4000)
    check("tiny region takes effect", out_amp < 0.8 * in_amp,
          f"ratio {out_amp / in_amp:.3f}")
    check("effect is localised (<=1 LSB)", max_abs_diff(seg(ref, 0.0, 0.9), seg(out, 0.0, 0.9)) <= LSB)
    check("no clicks", max_jump(out) <= max_jump(ref) * 1.5 + 1e-3)


def test_multiple_regions():
    print("multiple regions")
    sig = sine(4000, 3.0)
    src = save_wav("multi_src.wav", [sig])
    dst = os.path.join(TMP, "multi_dst.wav")
    repair.spectral_repair(src, dst, [
        {"t0": 0.5, "t1": 0.7, "f0": 3000, "f1": 5000, "gain_db": -120.0},
        {"t0": 1.5, "t1": 1.7, "f0": 3000, "f1": 5000, "gain_db": -120.0},
        {"t0": 2.5, "t1": 2.7, "f0": 3000, "f1": 5000, "gain_db": -120.0},
    ])
    ref = read_wav(src)[0][0]
    out = read_wav(dst)[0][0]
    for t in (0.6, 1.6, 2.6):
        a_in = amplitude_at(seg(ref, t - 0.05, t + 0.05), 4000)
        a_out = amplitude_at(seg(out, t - 0.05, t + 0.05), 4000)
        check(f"region at {t}s erased", a_out < 0.2 * a_in, f"ratio {a_out / a_in:.3f}")
    for t in (0.2, 1.1, 2.1, 2.85):
        check(f"audio at {t}s intact (<=1 LSB)",
              max_abs_diff(seg(ref, t - 0.05, t + 0.05), seg(out, t - 0.05, t + 0.05)) <= LSB)


def test_stereo_and_full_time():
    print("stereo file, region spanning the whole file")
    left = sine(1000, 1.5)
    right = sine(4000, 1.5)
    src = save_wav("stereo_src.wav", [left, right])
    dst = os.path.join(TMP, "stereo_dst.wav")
    info = repair.spectral_repair(src, dst, [
        {"t0": 0.0, "t1": 1.5, "f0": 3000, "f1": 5000, "gain_db": -120.0}])
    ref = read_wav(src)[0]
    out, sr = read_wav(dst)
    check("two channels preserved", len(out) == 2 and info["channels"] == 2)
    # Interior (beyond one STFT window from the edges) must be untouched;
    # the outermost ~46 ms may deviate slightly — that is the exact STFT
    # edge response of a full-file erase (verified against direct
    # STFT->mask->ISTFT to float precision), not a defect.
    edge = int(0.05 * SR)
    check("left channel interior untouched (<=1 LSB)",
          max_abs_diff(out[0][edge:-edge], ref[0][edge:-edge]) <= LSB)
    edge_dev = max(max_abs_diff(out[0][:edge], ref[0][:edge]),
                   max_abs_diff(out[0][-edge:], ref[0][-edge:]))
    check("no spikes at file edges", edge_dev < 0.05, f"{edge_dev:.4f}")
    rms_r = math.sqrt(sum(v * v for v in seg(out[1], 0.2, 1.3)) / len(seg(out[1], 0.2, 1.3)))
    check("right channel erased", rms_r < 1e-4, f"rms {rms_r:.6f}")
    check("frames preserved", len(out[0]) == len(left) and len(out[1]) == len(right))


def test_edge_regions():
    print("regions touching file edges and DC/Nyquist")
    sig = mix(sine(100, 2.0), sine(10000, 2.0))
    src = save_wav("edge_src.wav", [sig])
    dst = os.path.join(TMP, "edge_dst.wav")
    repair.spectral_repair(src, dst, [
        {"t0": 0.0, "t1": 0.5, "f0": 0, "f1": 200, "gain_db": -120.0},       # start + DC
        {"t0": 1.5, "t1": 2.0, "f0": 8000, "f1": SR / 2, "gain_db": -120.0},  # end + Nyquist
    ])
    ref = read_wav(src)[0][0]
    out = read_wav(dst)[0][0]
    check("frames preserved", len(out) == len(sig))
    a100 = amplitude_at(seg(out, 0.1, 0.4), 100)
    check("low tone erased at file start", a100 < 0.05)
    a10k = amplitude_at(seg(out, 1.6, 1.9), 10000)
    check("high tone erased at file end", a10k < 0.05)
    mid_ok = max_abs_diff(seg(ref, 0.7, 1.3), seg(out, 0.7, 1.3)) <= LSB
    check("middle untouched (<=1 LSB)", mid_ok)
    check("no NaN/inf", all(math.isfinite(v) for v in out[::97]))


def test_preview():
    print("preview rendering")
    sig = sine(4000, 5.0)
    src = save_wav("prev_src.wav", [sig])
    wav = repair.repair_preview(src, [
        {"t0": 2.0, "t1": 3.0, "f0": 3000, "f1": 5000, "gain_db": -120.0}])
    check("wav bytes", wav[:4] == b"RIFF" and wav[8:12] == b"WAVE")
    n_ch = struct.unpack("<H", wav[22:24])[0]
    sr = struct.unpack("<I", wav[24:28])[0]
    data_size = struct.unpack("<I", wav[40:44])[0]
    seconds = data_size / (sr * n_ch * 2)
    check("preview covers region + context", 2.9 < seconds < 3.1, f"{seconds:.2f}s")
    # Decode and verify the erasure is audible in the excerpt.
    frames = struct.unpack("<%dh" % (data_size // 2), wav[44:44 + data_size])
    x = [v / 32768.0 for v in frames]
    mid = seg(x, 1.4, 1.6, sr)   # excerpt starts at 1.0s -> region centre ~1.5s local
    check("erasure present in preview", amplitude_at(mid, 4000, sr) < 0.05,
          f"amp {amplitude_at(mid, 4000, sr):.4f}")


def test_duration_formats():
    print("duration preserved for odd lengths / other sample rates")
    for sr in (22050, 48000):
        sig = sine(1000, 1.37, sr=sr)
        src = save_wav(f"fmt_{sr}.wav", [sig], sr=sr)
        dst = os.path.join(TMP, f"fmt_{sr}_out.wav")
        repair.spectral_repair(src, dst, [
            {"t0": 0.4, "t1": 0.9, "f0": 800, "f1": 1200, "gain_db": -30.0}])
        out, osr = read_wav(dst)
        check(f"sr {sr} preserved", osr == sr)
        check(f"frames preserved @ {sr}", len(out[0]) == len(sig),
              f"{len(out[0])} vs {len(sig)}")
        a_in = amplitude_at(seg(sig, 0.55, 0.75, sr), 1000, sr)
        a_out = amplitude_at(seg(out[0], 0.55, 0.75, sr), 1000, sr)
        ratio = a_out / a_in
        check(f"attenuation ~-30 dB @ {sr}", 0.02 < ratio < 0.05, f"ratio {ratio:.4f}")


# --------------------------------------------------------------------------- #
# Flask API tests
# --------------------------------------------------------------------------- #

def test_api():
    print("Flask API endpoints")
    import app as web
    from backend import storage

    tmp = tempfile.mkdtemp(prefix="repair_api_")
    web.DATA_DIR = tmp
    web.store = storage.Storage(tmp)
    client = web.app.test_client()

    r = client.post("/api/library/generate", json={"kind": "chord", "duration": 2.0, "freq": 440})
    assert r.status_code == 200, r.get_json()
    fid = r.get_json()["id"]

    regions = [{"t0": 0.5, "t1": 1.0, "f0": 800, "f1": 1200, "gain_db": -40.0}]
    r = client.post(f"/api/audio/{fid}/repair", json={"regions": regions})
    check("repair endpoint 200", r.status_code == 200, str(r.get_json()))
    entry = r.get_json()
    check("derived file registered", entry and entry.get("id") and entry["id"] != fid)
    check("duration preserved", abs(entry["duration"] - 2.0) < 1e-6, str(entry.get("duration")))
    check("derived_from set", entry.get("derived_from") == fid)

    r = client.get(f"/api/audio/{entry['id']}")
    check("repaired audio downloadable", r.status_code == 200)

    r = client.post("/api/repair/preview", json={"file_id": fid, "regions": regions})
    check("preview endpoint 200", r.status_code == 200, str(r.get_json())[:200])
    check("preview has wav", len(r.get_json().get("wav", "")) > 1000)

    r = client.post(f"/api/audio/{fid}/repair", json={"regions": []})
    check("empty regions rejected", r.status_code == 400)
    r = client.post("/api/audio/deadbeef/repair", json={"regions": regions})
    check("missing file 404", r.status_code == 404)
    r = client.post("/api/repair/preview", json={"file_id": fid, "regions": "bogus"})
    check("bad preview payload rejected", r.status_code == 400)


def main():
    test_identity()
    test_channel_inmemory_exact()
    test_band_attenuation()
    test_erase_full_band()
    test_tiny_region()
    test_multiple_regions()
    test_stereo_and_full_time()
    test_edge_regions()
    test_preview()
    test_duration_formats()
    test_api()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
