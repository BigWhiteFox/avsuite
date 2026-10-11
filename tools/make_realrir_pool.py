"""Build real-RIR pool (16k mono wav) from downloaded raw sources for eval_v6.

Sources:
  - IR64-CAR (Zenodo 15168578, CC BY 4.0): Peugeot 208 cabin IRs, 48k/24bit,
    8 directions x 64 capsules, 1 s each. Extract 4 spatially spread capsules
    per direction -> 32 car-cabin RIRs, resampled to 16k.
  - OpenSLR SLR28 (Apache 2.0) real RIR subsets (RWCP / REVERB / AIR): any
    16k mono wavs found under D:/data/rir_real/openslr28/extracted are linked
    into the pool (already 16k).

Usage:
  python make_realrir_pool.py --car-dir D:/data/rir_real/ir64car/extracted \
      --out-dir D:/data/rir_real/pool_16k_mono
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
CAR_CAPSULES = [0, 8, 16, 24]  # spatially spread over the EM64 sphere


def resample_poly16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return x
    from scipy.signal import resample_poly
    g = np.gcd(sr_in, SR)
    return resample_poly(x, SR // g, sr_in // g)


def normalize_peak(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for wav in sorted(args.car_dir.glob("IR64CAR_*.wav")):
        data, sr = sf.read(str(wav), dtype="float64", always_2d=True)
        for cap in CAR_CAPSULES:
            if cap >= data.shape[1]:
                continue
            x = resample_poly16k(data[:, cap], sr)
            # trim to 300 ms ceiling like eval recipe (rir_max_ms also applies)
            x = x[: int(SR * 0.6)]  # keep a little headroom; synth truncates again
            x = normalize_peak(x)
            out = args.out_dir / f"car_{wav.stem}_c{cap:02d}.wav"
            sf.write(str(out), x, SR, subtype="PCM_16")
            n += 1
            print(f" wrote {out.name}  sr={SR} len={len(x)/SR:.2f}s")
    print(f"car RIRs: {n}")


if __name__ == "__main__":
    main()
