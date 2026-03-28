#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validação fisiológica simples (stress) por sessão, usando cliques manuais no Biopac
+ cache, e guardando tudo dentro de:

  analysis/<sessão>/stress_validation/

Entrada (por sessão):
  analysis/<sessão>/biopac/ecg.csv
  analysis/<sessão>/biopac/respiration.csv

Cache (por sessão):
  analysis/<sessão>/analysis_manual_markers.json

Saída (por sessão):
  analysis/<sessão>/stress_validation/per_session_metrics.csv
  analysis/<sessão>/stress_validation/summary.png

Consola:
  imprime tabela com valores absolutos + deltas (Δ) vs BASELINE

Requisitos:
  pip install numpy pandas matplotlib scipy
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks


# =========================
# Paths
# =========================

ANALYSIS_BASE_DIR = Path(r"C:/Users/david/Desktop/PASAT_Experiments/analysis")

MANUAL_MARKERS_FILENAME = "analysis_manual_markers.json"
OUTPUT_FOLDERNAME = "stress_validation"


# =========================
# Experiment constants
# =========================

PATTERN_DURATION = 23.0
DEFAULT_PHASES = ["BASELINE", "PASAT1", "PASAT2", "PASAT3", "RECOVERY"]

# ECG
ECG_BANDPASS = (5.0, 20.0)
FILTER_ORDER = 4
MIN_RR_SECONDS = 0.30
MAX_RR_SECONDS = 2.00
PROM_Q75_SCALE = 0.50

# Respiração
RESP_MIN_BREATH_SECONDS = 1.0
RESP_MAX_BREATH_SECONDS = 10.0

# Robustez / QC
SUBWINDOW_SECONDS = 30.0
MIN_BEATS_PER_SUBWINDOW = 8
MIN_RR_VALID_FRAC = 0.80
HR_PLAUSIBLE_RANGE = (40.0, 200.0)


# =========================
# Data classes
# =========================

@dataclass
class PhaseWindow:
    phase: str
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return float(self.t_end - self.t_start)


# =========================
# Session selection (from analysis/)
# =========================

def list_analysis_sessions() -> List[Path]:
    if not ANALYSIS_BASE_DIR.exists():
        raise FileNotFoundError(f"ANALYSIS_BASE_DIR não existe: {ANALYSIS_BASE_DIR}")

    sessions = sorted([p for p in ANALYSIS_BASE_DIR.iterdir() if p.is_dir()])
    usable = []
    for s in sessions:
        ecg_csv = s / "biopac" / "ecg.csv"
        resp_csv = s / "biopac" / "respiration.csv"
        if ecg_csv.exists() and resp_csv.exists():
            usable.append(s)

    if not usable:
        print("❌ Não encontrei sessões com analysis/<sessão>/biopac/ecg.csv e respiration.csv")
        return []

    print("\n" + "=" * 70)
    print("SESSÕES DISPONÍVEIS (analysis/)")
    print("=" * 70)
    for i, s in enumerate(usable):
        has_cache = (s / MANUAL_MARKERS_FILENAME).exists()
        cache_txt = " (cache OK)" if has_cache else ""
        print(f"[{i}] {s.name}{cache_txt}")
    print("=" * 70)
    return usable

def choose_session(sessions: List[Path]) -> Path | None:
    if not sessions:
        return None
    while True:
        raw = input("Seleciona a sessão (número) ou 'q' para sair: ").strip().lower()
        if raw == "q":
            return None
        try:
            idx = int(raw)
            if 0 <= idx < len(sessions):
                return sessions[idx]
        except ValueError:
            pass
        print("❌ Seleção inválida. Tenta novamente.")


# =========================
# CSV loaders
# =========================

def load_biopac_csvs(session_analysis_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_ecg, ecg, t_resp, resp
    """
    ecg_path = session_analysis_dir / "biopac" / "ecg.csv"
    resp_path = session_analysis_dir / "biopac" / "respiration.csv"

    ecg_df = pd.read_csv(ecg_path)
    resp_df = pd.read_csv(resp_path)

    t_ecg = ecg_df["time_seconds"].to_numpy(dtype=np.float64)
    if "ecg_mv" in ecg_df.columns:
        ecg = ecg_df["ecg_mv"].to_numpy(dtype=np.float64)
    else:
        col = [c for c in ecg_df.columns if c != "time_seconds"][0]
        ecg = ecg_df[col].to_numpy(dtype=np.float64)

    t_resp = resp_df["time_seconds"].to_numpy(dtype=np.float64)
    if "respiration" in resp_df.columns:
        resp = resp_df["respiration"].to_numpy(dtype=np.float64)
    else:
        col = [c for c in resp_df.columns if c != "time_seconds"][0]
        resp = resp_df[col].to_numpy(dtype=np.float64)

    return t_ecg, ecg, t_resp, resp


# =========================
# DSP helpers
# =========================

def estimate_fs_from_time(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time axis inválida para estimar fs.")
    return float(1.0 / dt)

def butter_bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)

def rr_intervals_seconds(peaks_idx: np.ndarray, fs: float) -> np.ndarray:
    if peaks_idx.size < 2:
        return np.array([], dtype=np.float64)
    return (np.diff(peaks_idx) / fs).astype(np.float64)

def mean_hr_bpm(rr_s: np.ndarray) -> float:
    if rr_s.size < 2:
        return np.nan
    return float(60.0 / np.mean(rr_s))

def rmssd(rr_s: np.ndarray) -> float:
    if rr_s.size < 3:
        return np.nan
    d = np.diff(rr_s)
    return float(np.sqrt(np.mean(d * d)))

def qc_rr(rr_s: np.ndarray) -> Tuple[float, float]:
    if rr_s.size == 0:
        return np.nan, np.nan
    valid = (rr_s >= MIN_RR_SECONDS) & (rr_s <= MAX_RR_SECONDS)
    return float(1.0 - np.mean(valid)), float(np.mean(valid))


# =========================
# ECG R-peak detection
# =========================

def detect_rpeaks(ecg: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, Dict]:
    fs = estimate_fs_from_time(t)

    mask = np.isfinite(ecg) & np.isfinite(t)
    ecg = ecg[mask]
    t = t[mask]

    ecg_f = butter_bandpass_filter(ecg, fs, ECG_BANDPASS[0], ECG_BANDPASS[1], order=FILTER_ORDER)
    energy = ecg_f * ecg_f

    prom = np.percentile(energy, 75) * PROM_Q75_SCALE
    prom = max(prom, 1e-12)

    min_distance = int(MIN_RR_SECONDS * fs)
    peaks, _ = find_peaks(energy, distance=min_distance, prominence=prom)

    debug = {"fs": fs, "prominence": float(prom), "n_peaks": int(peaks.size)}
    return peaks, debug


# =========================
# Respiration features
# =========================

def respiration_rate_bpm(resp_t: np.ndarray, resp: np.ndarray, start: float, end: float) -> float:
    m = (resp_t >= start) & (resp_t <= end)
    t = resp_t[m]
    x = resp[m]
    if t.size < 10:
        return np.nan

    fs = estimate_fs_from_time(t)
    win = max(3, int(0.3 * fs))
    x_s = np.convolve(x, np.ones(win) / win, mode="same")

    min_dist = int(RESP_MIN_BREATH_SECONDS * fs)
    peaks, _ = find_peaks(x_s, distance=min_dist)

    if peaks.size < 2:
        return np.nan

    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi >= RESP_MIN_BREATH_SECONDS) & (ibi <= RESP_MAX_BREATH_SECONDS)]
    if ibi.size == 0:
        return np.nan

    return float(60.0 / np.mean(ibi))


# =========================
# Manual markers cache + selection UI (on respiration)
# =========================

def manual_markers_path(session_analysis_dir: Path) -> Path:
    return session_analysis_dir / MANUAL_MARKERS_FILENAME

def load_cached_manual_markers(session_analysis_dir: Path) -> Dict[str, float] | None:
    p = manual_markers_path(session_analysis_dir)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    markers = data.get("markers", None)
    if not isinstance(markers, dict):
        return None
    return {k: float(v) for k, v in markers.items()}

def save_cached_manual_markers(session_analysis_dir: Path, session_name: str, markers: Dict[str, float]) -> None:
    p = manual_markers_path(session_analysis_dir)
    payload = {"session": session_name, "pattern_duration_s": PATTERN_DURATION, "markers": markers}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def pick_5_pattern_starts_on_resp(t: np.ndarray, resp: np.ndarray, session_name: str) -> Dict[str, float]:
    phase_names = [
        "GROUNDTRUTH_START",
        "PASAT1_START",
        "PASAT2_START",
        "PASAT3_START",
        "GROUNDTRUTH_FINAL_START",
    ]
    colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    if t.size > 200_000:
        step = t.size // 80_000
        t_plot = t[::step]
        resp_plot = resp[::step]
    else:
        t_plot = t
        resp_plot = resp

    selected: List[float] = []

    for i, name in enumerate(phase_names):
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(t_plot, resp_plot, lw=0.8, alpha=0.75, color="#27ae60", label="Respiração (Biopac)")

        for j, prev in enumerate(selected):
            ax.axvline(prev, color=colors[j], ls="--", lw=2, alpha=0.9, label=f"{phase_names[j]} @ {prev:.2f}s")
            ax.axvspan(prev, prev + PATTERN_DURATION, color=colors[j], alpha=0.08)

        ax.set_title(f"{session_name} — clica no INÍCIO do padrão: {name}", fontweight="bold", color=colors[i])
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Respiração")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

        clicked = {"t": None}

        def onclick(event):
            if event.xdata is None:
                return
            clicked["t"] = float(event.xdata)
            ax.axvline(clicked["t"], color="red", lw=3)
            ax.axvspan(clicked["t"], clicked["t"] + PATTERN_DURATION, color="red", alpha=0.10)
            fig.canvas.draw_idle()
            print(f"✅ Selecionado {name}: {clicked['t']:.3f}s (fecha a janela para confirmar)")

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.tight_layout()
        plt.show()
        fig.canvas.mpl_disconnect(cid)

        if clicked["t"] is None:
            raise RuntimeError(f"Seleção cancelada/sem clique para {name}.")

        selected.append(clicked["t"])

    return dict(zip(phase_names, selected))


# =========================
# Phase windows
# =========================

def infer_mode_and_durations(total_duration_s: float) -> Tuple[str, float, float]:
    if total_duration_s > 5 * 60:
        return "REAL", 300.0, 150.0
    return "TESTE", 20.0, 20.0

def build_phase_windows(markers: Dict[str, float], total_duration_s: float) -> Tuple[str, List[PhaseWindow]]:
    mode, gt_dur, pasat_dur = infer_mode_and_durations(total_duration_s)

    windows = [
        PhaseWindow("BASELINE", markers["GROUNDTRUTH_START"] + PATTERN_DURATION,
                    markers["GROUNDTRUTH_START"] + PATTERN_DURATION + gt_dur),
        PhaseWindow("PASAT1", markers["PASAT1_START"] + PATTERN_DURATION,
                    markers["PASAT1_START"] + PATTERN_DURATION + pasat_dur),
        PhaseWindow("PASAT2", markers["PASAT2_START"] + PATTERN_DURATION,
                    markers["PASAT2_START"] + PATTERN_DURATION + pasat_dur),
        PhaseWindow("PASAT3", markers["PASAT3_START"] + PATTERN_DURATION,
                    markers["PASAT3_START"] + PATTERN_DURATION + pasat_dur),
        PhaseWindow("RECOVERY", markers["GROUNDTRUTH_FINAL_START"] + PATTERN_DURATION,
                    markers["GROUNDTRUTH_FINAL_START"] + PATTERN_DURATION + gt_dur),
    ]
    return mode, windows


# =========================
# Subwindow aggregation
# =========================

def iter_subwindows(w: PhaseWindow, step_s: float) -> List[Tuple[float, float]]:
    starts = np.arange(w.t_start, w.t_end, step_s)
    subs = []
    for s in starts:
        e = min(s + step_s, w.t_end)
        if e - s >= max(10.0, 0.5 * step_s):
            subs.append((float(s), float(e)))
    return subs

def window_metrics_from_peaks(ecg_t: np.ndarray, peaks_idx: np.ndarray, fs: float, start: float, end: float) -> Dict:
    peaks_t = ecg_t[np.clip(peaks_idx, 0, ecg_t.size - 1)]
    in_w = (peaks_t >= start) & (peaks_t <= end)
    p = peaks_idx[in_w]
    rr = rr_intervals_seconds(p, fs)

    hr = mean_hr_bpm(rr)
    r = rmssd(rr)
    _, rr_valid_frac = qc_rr(rr)

    return {"n_beats": int(p.size), "hr_bpm": hr, "rmssd_s": r, "rr_valid_frac": rr_valid_frac}

def is_good_subwindow(m: Dict) -> bool:
    if not np.isfinite(m["hr_bpm"]) or not np.isfinite(m["rmssd_s"]) or not np.isfinite(m["rr_valid_frac"]):
        return False
    if m["n_beats"] < MIN_BEATS_PER_SUBWINDOW:
        return False
    if m["rr_valid_frac"] < MIN_RR_VALID_FRAC:
        return False
    if not (HR_PLAUSIBLE_RANGE[0] <= m["hr_bpm"] <= HR_PLAUSIBLE_RANGE[1]):
        return False
    return True

def aggregate_phase_metrics(ecg_t: np.ndarray, peaks_idx: np.ndarray, fs: float,
                            resp_t: np.ndarray, resp: np.ndarray, w: PhaseWindow) -> Dict:
    subs = iter_subwindows(w, SUBWINDOW_SECONDS)
    good_flags = []
    good_metrics = []

    for (s, e) in subs:
        m = window_metrics_from_peaks(ecg_t, peaks_idx, fs, s, e)
        good = is_good_subwindow(m)
        good_flags.append(good)
        if good:
            good_metrics.append(m)

    valid_window_frac = float(np.mean(good_flags)) if good_flags else np.nan

    if good_metrics:
        hr = float(np.nanmedian([m["hr_bpm"] for m in good_metrics]))
        rm = float(np.nanmedian([m["rmssd_s"] for m in good_metrics]))
        rr_valid_med = float(np.nanmedian([m["rr_valid_frac"] for m in good_metrics]))
        nbeats_used = int(np.sum([m["n_beats"] for m in good_metrics]))
    else:
        hr = np.nan
        rm = np.nan
        rr_valid_med = np.nan
        nbeats_used = 0

    rrate = respiration_rate_bpm(resp_t, resp, w.t_start, w.t_end)

    return {
        "phase": w.phase,
        "t_start": w.t_start,
        "t_end": w.t_end,
        "duration_s": w.duration,
        "subwindow_s": SUBWINDOW_SECONDS,
        "valid_window_frac": valid_window_frac,
        "n_beats_used": nbeats_used,
        "hr_bpm": hr,
        "rmssd_s": rm,
        "rr_valid_frac_med": rr_valid_med,
        "resp_rate_bpm": rrate,
    }


# =========================
# Printing + plots
# =========================

def print_metrics_table(session_name: str, mode: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df["phase"] = pd.Categorical(df["phase"], categories=DEFAULT_PHASES, ordered=True)
    df = df.sort_values("phase")

    # valores absolutos
    show = df[[
        "phase", "duration_s", "hr_bpm", "rmssd_s", "resp_rate_bpm",
        "valid_window_frac", "n_beats_used"
    ]].copy()

    # baseline para deltas
    baseline = df[df["phase"] == "BASELINE"]
    if not baseline.empty:
        hr0 = float(baseline["hr_bpm"].iloc[0])
        rm0 = float(baseline["rmssd_s"].iloc[0])
        rr0 = float(baseline["resp_rate_bpm"].iloc[0])
    else:
        hr0 = rm0 = rr0 = np.nan

    # deltas vs baseline
    show["d_hr_bpm"] = show["hr_bpm"] - hr0
    show["d_rmssd_s"] = show["rmssd_s"] - rm0
    show["d_resp_rate_bpm"] = show["resp_rate_bpm"] - rr0

    def f_num(x, fmt):
        return fmt.format(x) if np.isfinite(x) else "NaN"

    show["duration_s"] = show["duration_s"].map(lambda x: f_num(x, "{:.0f}"))
    show["hr_bpm"] = show["hr_bpm"].map(lambda x: f_num(x, "{:.1f}"))
    show["rmssd_s"] = show["rmssd_s"].map(lambda x: f_num(x, "{:.4f}"))
    show["resp_rate_bpm"] = show["resp_rate_bpm"].map(lambda x: f_num(x, "{:.2f}"))
    show["valid_window_frac"] = show["valid_window_frac"].map(lambda x: f_num(x, "{:.2f}"))

    show["d_hr_bpm"] = show["d_hr_bpm"].map(lambda x: f_num(x, "{:+.1f}"))
    show["d_rmssd_s"] = show["d_rmssd_s"].map(lambda x: f_num(x, "{:+.4f}"))
    show["d_resp_rate_bpm"] = show["d_resp_rate_bpm"].map(lambda x: f_num(x, "{:+.2f}"))

    print("\n" + "=" * 70)
    print(f"STRESS VALIDATION — {session_name} [{mode}]")
    print("Valores absolutos + Δ vs BASELINE")
    print("=" * 70)
    print(show.to_string(index=False))
    print("=" * 70)

def plot_session_summary(session_name: str, mode: str, df: pd.DataFrame, out_path: Path) -> None:
    df = df.copy()
    df["phase"] = pd.Categorical(df["phase"], categories=DEFAULT_PHASES, ordered=True)
    df = df.sort_values("phase")

    fig, ax = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)

    ax[0].plot(df["phase"], df["hr_bpm"], marker="o")
    ax[0].set_title("HR"); ax[0].set_ylabel("bpm"); ax[0].grid(True, alpha=0.3)

    ax[1].plot(df["phase"], df["rmssd_s"], marker="o", color="#8e44ad")
    ax[1].set_title("RMSSD"); ax[1].set_ylabel("s"); ax[1].grid(True, alpha=0.3)

    ax[2].plot(df["phase"], df["resp_rate_bpm"], marker="o", color="#27ae60")
    ax[2].set_title("Resp rate"); ax[2].set_ylabel("breaths/min"); ax[2].grid(True, alpha=0.3)

    ax[3].plot(df["phase"], df["valid_window_frac"], marker="o", color="#2c3e50")
    ax[3].set_title("QC (janelas boas)"); ax[3].set_ylabel("0–1")
    ax[3].set_ylim(-0.05, 1.05); ax[3].grid(True, alpha=0.3)

    fig.suptitle(f"{session_name} — Summary [{mode}] (subwindows={SUBWINDOW_SECONDS:.0f}s)", fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =========================
# Main
# =========================

def main():
    sessions = list_analysis_sessions()
    if not sessions:
        return

    session_dir = choose_session(sessions)
    if session_dir is None:
        return

    session_name = session_dir.name
    out_dir = session_dir / OUTPUT_FOLDERNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 70)
    print("PROCESSANDO:", session_name)
    print("OUTPUT:", out_dir)
    print("-" * 70)

    # 1) carregar CSVs Biopac exportados
    t_ecg, ecg, t_resp, resp = load_biopac_csvs(session_dir)

    total_dur = float(t_ecg[-1] - t_ecg[0])
    print(f"ECG duração: {total_dur:.1f}s")

    # 2) cache ou cliques (no sinal de respiração)
    markers = load_cached_manual_markers(session_dir)
    if markers is None:
        print(f"🖱️  Sem cache ({MANUAL_MARKERS_FILENAME}). Seleciona 5 padrões…")
        markers = pick_5_pattern_starts_on_resp(t_resp, resp, session_name)
        save_cached_manual_markers(session_dir, session_name, markers)
        print(f"✅ Guardado: {manual_markers_path(session_dir)}")
    else:
        print(f"✅ Usando cache: {manual_markers_path(session_dir)}")

    # 3) janelas por fase
    mode, windows = build_phase_windows(markers, total_dur)
    print("Modo inferido:", mode)

    # 4) R-peaks
    peaks_idx, dbg = detect_rpeaks(ecg, t_ecg)
    fs = dbg["fs"]
    print(f"R-peaks: {dbg['n_peaks']} | fs≈{fs:.1f} Hz | prom≈{dbg['prominence']:.3g}")

    # 5) métricas robustas
    rows = []
    for w in windows:
        m = aggregate_phase_metrics(t_ecg, peaks_idx, fs, t_resp, resp, w)
        m["session"] = session_name
        m["mode"] = mode
        rows.append(m)

    df_sess = pd.DataFrame(rows)

    # 6) print tabela (com deltas)
    print_metrics_table(session_name, mode, df_sess)

    # 7) guardar CSV + plot
    csv_path = out_dir / "per_session_metrics.csv"
    img_path = out_dir / "summary.png"

    df_sess.to_csv(csv_path, index=False)
    plot_session_summary(session_name, mode, df_sess, img_path)

    print("\n✅ Guardado:")
    print(" -", csv_path)
    print(" -", img_path)


if __name__ == "__main__":
    main()