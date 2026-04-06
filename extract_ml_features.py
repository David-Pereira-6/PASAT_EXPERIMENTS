#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_ml_features.py

Generate per-session ML features (one row per 30s window, 50% overlap / 15s step)
for PASAT-C stress classification.

Inputs (per session):
- analysis/<session>/usrp/data.csv
    columns: time_seconds, phase_demodulated_radians (optionally magnitude)
- analysis/<session>/analysis_manual_markers.json
    format:
      {"pattern_duration_s": 23.0,
       "markers": {"GROUNDTRUTH_START": ..., "PASAT1_START": ..., ...}}
- sessions/<session>/xenics/npy/frame_XXXX.npy (16-bit frames, 0..65535)

Outputs (per session):
- analysis/<session>/ml_features/features.csv

This script is intentionally NON-interactive and produces no plots.

Notes:
- Uses the synchronized timeline from analysis/ markers (already aligned with Biopac anchor).
- Xenics frames are mapped using FPS=13: frame_idx ~= round(t_seconds * FPS)
- Xenics features are patch-based (robust to cuts). ROI/landmark features can be added later.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# --------------------------
# Configuration / constants
# --------------------------

FPS_XENICS = 13.0
PATTERN_DURATION_DEFAULT = 23.0
PHASE_DURATIONS_DEFAULT = {"groundtruth": 300.0, "pasat": 150.0}

WINDOW_SECONDS = 30.0
WINDOW_OVERLAP = 0.5
WINDOW_STEP_SECONDS = WINDOW_SECONDS * (1.0 - WINDOW_OVERLAP)  # 15s

# BioRadar bands
BR_HR_BAND_HZ = (1.2, 3.0)  # 72-180 bpm
BR_RR_BAND_HZ = (0.2, 0.6)  # 12-36 rpm

FFT_ZERO_PADDING_FACTOR = 4

# Xenics frame processing
XENICS_MAX_VALUE_16BIT = 65535.0
XENICS_8BIT_BINS = 32

# Efficiency: sample frames inside each 30s window (don’t load all 390 frames)
XENICS_MAX_FRAMES_PER_WINDOW = 60  # ~ 2 fps sampling


# --------------------------
# Small utilities
# --------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _estimate_fs_from_time(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    return float(1.0 / dt)


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _preprocess_1d_signal(x: np.ndarray, fs: float, window_percent: float = 0.1) -> np.ndarray:
    """Smooth + z-normalize (robust baseline for FFT)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 5 or not np.isfinite(fs) or fs <= 0:
        return x

    win = max(3, int(fs * window_percent))
    x_s = uniform_filter1d(x, size=win, mode="nearest")
    x_c = x_s - np.mean(x_s)
    sd = np.std(x_c)
    if sd > 0:
        x_c = x_c / sd
    return x_c


def _fft_band_features(
    x: np.ndarray,
    fs: float,
    band_hz: Tuple[float, float],
) -> Dict[str, float]:
    """Return peak bpm, peak mag, snr, top2 ratio and approximate peak width."""
    out = {
        "bpm": np.nan,
        "peak_mag": np.nan,
        "snr": np.nan,
        "top2_ratio": np.nan,
        "peak_width_hz": np.nan,
    }
    if x.size < 10 or not np.isfinite(fs) or fs <= 0:
        return out

    # Windowing + zero-padding
    w = np.hanning(x.size)
    xw = x * w
    n_fft = int(xw.size * FFT_ZERO_PADDING_FACTOR)
    fft = np.fft.rfft(xw, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mag = np.abs(fft)

    m = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(m):
        return out

    f = freqs[m]
    y = mag[m]
    if y.size == 0 or np.all(~np.isfinite(y)):
        return out

    # Peak detection (fallback to argmax)
    y_max = float(np.nanmax(y))
    if not np.isfinite(y_max) or y_max <= 0:
        return out

    peaks, props = find_peaks(y, prominence=y_max * 0.05)
    if peaks.size == 0:
        peaks = np.array([int(np.nanargmax(y))])

    # Sort peaks by magnitude
    peaks = peaks[np.argsort(y[peaks])[::-1]]
    p0 = int(peaks[0])

    peak_freq = float(f[p0])
    peak_mag = float(y[p0])

    # SNR proxy: peak / median band magnitude
    med_band = float(np.median(y[np.isfinite(y)]))
    snr = peak_mag / med_band if med_band > 0 else np.nan

    # Top2 ratio
    if peaks.size >= 2:
        p1 = int(peaks[1])
        top2_ratio = peak_mag / float(y[p1]) if float(y[p1]) > 0 else np.nan
    else:
        top2_ratio = np.nan

    # Approx peak width: full-width at half max (within band)
    half = peak_mag * 0.5
    left = p0
    while left > 0 and y[left] >= half:
        left -= 1
    right = p0
    while right < y.size - 1 and y[right] >= half:
        right += 1
    peak_width_hz = float(f[right] - f[left]) if right > left else np.nan

    out.update(
        {
            "bpm": float(peak_freq * 60.0),
            "peak_mag": peak_mag,
            "snr": float(snr) if np.isfinite(snr) else np.nan,
            "top2_ratio": float(top2_ratio) if np.isfinite(top2_ratio) else np.nan,
            "peak_width_hz": peak_width_hz,
        }
    )
    return out


# --------------------------
# Xenics patch features
# --------------------------

def _frame_to_8bit(frame_u16: np.ndarray) -> np.ndarray:
    x = frame_u16.astype(np.float32) / XENICS_MAX_VALUE_16BIT * 255.0
    return np.clip(x, 0.0, 255.0).astype(np.float32)


def _central_patch(frame_8: np.ndarray) -> np.ndarray:
    h, w = frame_8.shape[:2]
    hs, he = h // 3, (2 * h) // 3
    ws, we = w // 3, (2 * w) // 3
    return frame_8[hs:he, ws:we]


def _entropy_32bins(x: np.ndarray) -> float:
    x = x[np.isfinite(x)].astype(np.float64)
    if x.size == 0:
        return float("nan")
    hist, _ = np.histogram(x, bins=XENICS_8BIT_BINS, range=(0.0, 255.0), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return float("nan")
    return float(-np.sum(hist * np.log2(hist)))


def _gradmag_mean(patch: np.ndarray) -> float:
    # simple gradient magnitude (no cv2 dependency)
    gy = np.diff(patch, axis=0, prepend=patch[:1, :])
    gx = np.diff(patch, axis=1, prepend=patch[:, :1])
    g = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(g))


def _sharpness_laplacian_var(patch: np.ndarray) -> float:
    # discrete laplacian
    c = patch
    up = np.roll(c, -1, axis=0)
    down = np.roll(c, 1, axis=0)
    left = np.roll(c, 1, axis=1)
    right = np.roll(c, -1, axis=1)
    lap = -4.0 * c + up + down + left + right
    return float(np.var(lap))


def _lin_slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    y = y[m]
    if y.size < 3:
        return float("nan")
    t = np.arange(y.size, dtype=np.float64)
    t = t - np.mean(t)
    y = y - np.mean(y)
    denom = float(np.sum(t * t))
    if denom <= 0:
        return float("nan")
    return float(np.sum(t * y) / denom)


def _xenics_window_features(
    frames_dir: Path,
    t_start: float,
    t_end: float,
    fps: float = FPS_XENICS,
) -> Dict[str, float]:
    """Patch-based Xenics features for a time window.

    Maps timeline seconds -> frame indices using fps.
    Samples up to XENICS_MAX_FRAMES_PER_WINDOW frames evenly.
    """
    out = {
        # QC / context
        "xn_mode": "patch_only",
        "xn_face_coverage": np.nan,  # placeholder (no face mask yet)
        "xn_motion_energy": np.nan,
        "xn_sharpness": np.nan,
        "xn_global_mean": np.nan,
        "xn_global_std": np.nan,
        # Patch global
        "xn_patch_median_mean": np.nan,
        "xn_patch_iqr_mean": np.nan,
        "xn_patch_entropy_mean": np.nan,
        "xn_patch_gradmag_mean": np.nan,
        "xn_patch_slope": np.nan,
        "xn_patch_std_t": np.nan,
        "xn_patch_mean_abs_dt": np.nan,
    }

    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        return out

    # Convert to frame indices
    i0 = int(round(t_start * fps))
    i1 = int(round(t_end * fps))
    if i1 <= i0:
        return out

    frame_indices = np.arange(i0, i1 + 1)
    if frame_indices.size == 0:
        return out

    # Subsample indices for speed
    if frame_indices.size > XENICS_MAX_FRAMES_PER_WINDOW:
        frame_indices = np.linspace(frame_indices[0], frame_indices[-1], XENICS_MAX_FRAMES_PER_WINDOW)
        frame_indices = np.unique(np.round(frame_indices).astype(int))

    patch_medians: List[float] = []
    patch_iqrs: List[float] = []
    entropies: List[float] = []
    gradmags: List[float] = []
    sharpness: List[float] = []

    prev_patch_med: Optional[float] = None
    motion_acc = []

    global_means = []
    global_stds = []

    last_frame_8: Optional[np.ndarray] = None

    for idx in frame_indices:
        fpath = frames_dir / f"frame_{idx:04d}.npy"
        if not fpath.exists():
            continue
        try:
            frame_u16 = np.load(fpath)
        except Exception:
            continue
        if frame_u16.ndim != 2:
            continue

        frame_8 = _frame_to_8bit(frame_u16)

        # Global stats (on central patch as robust substitute)
        patch = _central_patch(frame_8)
        if patch.size == 0:
            continue

        global_means.append(float(np.mean(patch)))
        global_stds.append(float(np.std(patch)))

        med = float(np.median(patch))
        patch_medians.append(med)
        patch_iqrs.append(float(np.percentile(patch, 75) - np.percentile(patch, 25)))
        entropies.append(_entropy_32bins(patch))
        gradmags.append(_gradmag_mean(patch))
        sharpness.append(_sharpness_laplacian_var(patch))

        # Motion energy (difference between consecutive sampled frames)
        if last_frame_8 is not None and last_frame_8.shape == frame_8.shape:
            d = np.abs(frame_8 - last_frame_8)
            # Use central patch diff as robust motion estimate
            dp = _central_patch(d)
            motion_acc.append(float(np.mean(dp)))
        last_frame_8 = frame_8

    if len(patch_medians) < 3:
        return out

    patch_medians_np = np.asarray(patch_medians, dtype=np.float64)

    out["xn_motion_energy"] = float(np.nanmean(motion_acc)) if motion_acc else np.nan
    out["xn_sharpness"] = float(np.nanmean(sharpness)) if sharpness else np.nan

    out["xn_global_mean"] = float(np.nanmean(global_means)) if global_means else np.nan
    out["xn_global_std"] = float(np.nanmean(global_stds)) if global_stds else np.nan

    out["xn_patch_median_mean"] = float(np.nanmean(patch_medians_np))
    out["xn_patch_iqr_mean"] = float(np.nanmean(patch_iqrs)) if patch_iqrs else np.nan
    out["xn_patch_entropy_mean"] = float(np.nanmean(entropies)) if entropies else np.nan
    out["xn_patch_gradmag_mean"] = float(np.nanmean(gradmags)) if gradmags else np.nan

    out["xn_patch_slope"] = _lin_slope(patch_medians_np)
    out["xn_patch_std_t"] = float(np.nanstd(patch_medians_np))
    out["xn_patch_mean_abs_dt"] = float(np.nanmean(np.abs(np.diff(patch_medians_np))))

    return out


# --------------------------
# Phase/window logic
# --------------------------

dataclass Phase:
    name: str
    t_start: float
    t_end: float


def _load_manual_markers(path: Path) -> Tuple[float, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    pattern_duration = float(j.get("pattern_duration_s", PATTERN_DURATION_DEFAULT))
    markers = j.get("markers", {})
    if not isinstance(markers, dict):
        raise ValueError("analysis_manual_markers.json: expected markers dict")
    markers_f = {k: float(v) for k, v in markers.items()}
    return pattern_duration, markers_f


def _build_phases(markers: Dict[str, float], pattern_duration: float) -> List[Phase]:
    gt = PHASE_DURATIONS_DEFAULT["groundtruth"]
    pasat = PHASE_DURATIONS_DEFAULT["pasat"]

    # markers are the start of the PATTERN; tests start after pattern_duration
    return [
        Phase(
            "baseline",
            markers["GROUNDTRUTH_START"] + pattern_duration,
            markers["GROUNDTRUTH_START"] + pattern_duration + gt,
        ),
        Phase(
            "pasat1",
            markers["PASAT1_START"] + pattern_duration,
            markers["PASAT1_START"] + pattern_duration + pasat,
        ),
        Phase(
            "pasat2",
            markers["PASAT2_START"] + pattern_duration,
            markers["PASAT2_START"] + pattern_duration + pasat,
        ),
        Phase(
            "pasat3",
            markers["PASAT3_START"] + pattern_duration,
            markers["PASAT3_START"] + pattern_duration + pasat,
        ),
        Phase(
            "recovery",
            markers["GROUNDTRUTH_FINAL_START"] + pattern_duration,
            markers["GROUNDTRUTH_FINAL_START"] + pattern_duration + gt,
        ),
    ]


def _iter_windows(phase: Phase) -> List[Tuple[float, float]]:
    starts = np.arange(phase.t_start, phase.t_end - WINDOW_SECONDS + 1e-9, WINDOW_STEP_SECONDS)
    windows = [(float(s), float(s + WINDOW_SECONDS)) for s in starts]

    # match your earlier approach: drop first and last window of each phase if possible
    if len(windows) > 2:
        windows = windows[1:-1]
    return windows


# --------------------------
# Main per-session extraction
# --------------------------

def _bioradar_window_features(df_usrp: pd.DataFrame, t_start: float, t_end: float) -> Dict[str, float]:
    out = {
        "br_fs_est": np.nan,
        "br_signal_std": np.nan,
        "br_signal_mad": np.nan,
        # HR
        "br_hr_fft_bpm": np.nan,
        "br_hr_fft_peak_mag": np.nan,
        "br_hr_fft_snr": np.nan,
        "br_hr_fft_top2_ratio": np.nan,
        "br_hr_fft_peak_width_hz": np.nan,
        # RR
        "br_rr_fft_bpm": np.nan,
        "br_rr_fft_peak_mag": np.nan,
        "br_rr_fft_snr": np.nan,
        "br_rr_fft_top2_ratio": np.nan,
        "br_rr_fft_peak_width_hz": np.nan,
        # QC
        "br_window_valid": 0.0,
    }

    m = (df_usrp["time_seconds"] >= t_start) & (df_usrp["time_seconds"] <= t_end)
    w = df_usrp.loc[m]
    if w.shape[0] < 10:
        return out

    t = w["time_seconds"].to_numpy(dtype=np.float64)
    x = w["phase_demodulated_radians"].to_numpy(dtype=np.float64)

    fs = _estimate_fs_from_time(t)
    out["br_fs_est"] = fs

    # preprocess
    x_p = _preprocess_1d_signal(x, fs, window_percent=0.1)

    out["br_signal_std"] = float(np.std(x_p)) if x_p.size else np.nan
    out["br_signal_mad"] = _mad(x_p)

    # FFT features
    hr = _fft_band_features(x_p, fs, BR_HR_BAND_HZ)
    rr = _fft_band_features(x_p, fs, BR_RR_BAND_HZ)

    out.update(
        {
            "br_hr_fft_bpm": hr["bpm"],
            "br_hr_fft_peak_mag": hr["peak_mag"],
            "br_hr_fft_snr": hr["snr"],
            "br_hr_fft_top2_ratio": hr["top2_ratio"],
            "br_hr_fft_peak_width_hz": hr["peak_width_hz"],
            "br_rr_fft_bpm": rr["bpm"],
            "br_rr_fft_peak_mag": rr["peak_mag"],
            "br_rr_fft_snr": rr["snr"],
            "br_rr_fft_top2_ratio": rr["top2_ratio"],
            "br_rr_fft_peak_width_hz": rr["peak_width_hz"],
        }
    )

    # Simple QC: plausible ranges + snr
    hr_ok = np.isfinite(out["br_hr_fft_bpm"]) and (40.0 <= out["br_hr_fft_bpm"] <= 200.0)
    rr_ok = np.isfinite(out["br_rr_fft_bpm"]) and (6.0 <= out["br_rr_fft_bpm"] <= 45.0)
    snr_ok = (np.isfinite(out["br_hr_fft_snr"]) and out["br_hr_fft_snr"] >= 2.0) or (
        np.isfinite(out["br_rr_fft_snr"]) and out["br_rr_fft_snr"] >= 2.0
    )
    out["br_window_valid"] = float(hr_ok and rr_ok and snr_ok)

    return out


def process_session(
    analysis_base_dir: Path,
    sessions_base_dir: Path,
    session: str,
) -> Path:
    sess_analysis = analysis_base_dir / session
    sess_sessions = sessions_base_dir / session

    usrp_csv = sess_analysis / "usrp" / "data.csv"
    markers_json = sess_analysis / "analysis_manual_markers.json"
    xenics_frames_dir = sess_sessions / "xenics" / "npy"

    if not usrp_csv.exists():
        raise FileNotFoundError(f"Missing USRP CSV: {usrp_csv}")
    if not markers_json.exists():
        raise FileNotFoundError(f"Missing manual markers: {markers_json}")
    if not xenics_frames_dir.exists():
        raise FileNotFoundError(f"Missing Xenics frames dir: {xenics_frames_dir}")

    pattern_duration, markers = _load_manual_markers(markers_json)
    phases = _build_phases(markers, pattern_duration)

    df_usrp = pd.read_csv(usrp_csv)
    required_cols = {"time_seconds", "phase_demodulated_radians"}
    if not required_cols.issubset(set(df_usrp.columns)):
        raise ValueError(f"USRP CSV missing columns. Required: {sorted(required_cols)}")

    out_dir = sess_analysis / "ml_features"
    _safe_mkdir(out_dir)

    rows: List[Dict[str, object]] = []

    for ph in phases:
        label_stress = 1 if ph.name.startswith("pasat") else 0
        windows = _iter_windows(ph)

        for w_idx, (t0, t1) in enumerate(windows):
            row: Dict[str, object] = {
                "session": session,
                "phase": ph.name,
                "label_stress": int(label_stress),
                "t_start": float(t0),
                "t_end": float(t1),
                "window_idx": int(w_idx),
            }

            # BioRadar
            row.update(_bioradar_window_features(df_usrp, t0, t1))

            # Xenics (patch only)
            row.update(_xenics_window_features(xenics_frames_dir, t0, t1, fps=FPS_XENICS))

            rows.append(row)

    df_out = pd.DataFrame(rows)

    out_csv = out_dir / "features.csv"
    df_out.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-window ML features (BioRadar + Xenics)")
    parser.add_argument(
        "--analysis-base-dir",
        type=str,
        default=os.getenv("ANALYSIS_BASE_DIR", ""),
        help="Root folder that contains analysis/<session>/... (can also use env ANALYSIS_BASE_DIR)",
    )
    parser.add_argument(
        "--sessions-base-dir",
        type=str,
        default=os.getenv("SESSIONS_BASE_DIR", ""),
        help="Root folder that contains sessions/<session>/... (can also use env SESSIONS_BASE_DIR)",
    )
    parser.add_argument("--session", type=str, default="", help="If set, process only this session")

    args = parser.parse_args()

    if not args.analysis_base_dir:
        raise SystemExit("Missing --analysis-base-dir (or env ANALYSIS_BASE_DIR)")
    if not args.sessions_base_dir:
        raise SystemExit("Missing --sessions-base-dir (or env SESSIONS_BASE_DIR)")

    analysis_base_dir = Path(args.analysis_base_dir)
    sessions_base_dir = Path(args.sessions_base_dir)

    if args.session:
        out = process_session(analysis_base_dir, sessions_base_dir, args.session)
        print(f"✅ Wrote: {out}")
        return

    # process all sessions under analysis/
    sessions = sorted([p.name for p in analysis_base_dir.iterdir() if p.is_dir()])
    for s in sessions:
        try:
            out = process_session(analysis_base_dir, sessions_base_dir, s)
            print(f"✅ {s}: {out}")
        except Exception as e:
            print(f"⚠️ {s}: {e}")


if __name__ == "__main__":
    main()