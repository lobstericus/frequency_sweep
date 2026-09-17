"""
Standalone plotting tool for frequency sweep experiments.

Loads analysis data from one or more experiment directories (each containing
an analysis/report.csv file produced by analyze.py) and generates combined
overlay plots saved to --output-dir:

  1. resonance_response.png   - transfer-function magnitude & phase
  2. rms_response.png         - RMS signal level vs frequency
  3. sorted_magnitude_response.png - sorted magnitude response

Usage:
  python plot.py --input-dir experiments/run1 --input-dir experiments/run2 \\
                 --output-dir plots/comparison

Each --input-dir is expected to contain either:
  <input-dir>/analysis/report.csv   (preferred)
  <input-dir>/report.csv            (legacy fallback)
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_experiment_csv(input_dir):
    """
    Load a report.csv from *input_dir* and return arrays suitable for plotting.

    Looks for the CSV at:
      1. <input_dir>/analysis/report.csv
      2. <input_dir>/report.csv

    Returns
    -------
    freqs, magnitudes, phases, piezo_dbfs, mic_dbfs : np.ndarray
    """
    candidates = [
        os.path.join(input_dir, "analysis", "report.csv"),
        os.path.join(input_dir, "report.csv"),
        os.path.join("experiments", input_dir, "analysis", "report.csv"),
        os.path.join("experiments", input_dir, "report.csv"),
    ]

    report_path = None
    for cand in candidates:
        if os.path.isfile(cand):
            report_path = cand
            break

    if not report_path:
        raise FileNotFoundError(
            f"report.csv not found for '{input_dir}'. Tried:\n" +
            "\n".join(f"  {c}" for c in candidates)
        )

    with open(report_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"report.csv is empty: {report_path}")

    freqs = np.array([float(row["frequency_hz"]) for row in rows], dtype=float)
    magnitudes = np.array([float(row["magnitude"]) for row in rows], dtype=float)
    phases = np.array([float(row["phase_radians"]) for row in rows], dtype=float)
    piezo_rms = np.array([float(row["piezo_rms"]) for row in rows], dtype=float)
    mic_rms = np.array([float(row["mic_rms"]) for row in rows], dtype=float)

    piezo_dbfs = np.array([
        20 * np.log10(v) if v > 0 else -np.inf for v in piezo_rms
    ])
    mic_dbfs = np.array([
        20 * np.log10(v) if v > 0 else -np.inf for v in mic_rms
    ])

    return freqs, magnitudes, phases, piezo_dbfs, mic_dbfs


def load_band_magnitudes_csv(input_dir):
    """
    Load band_magnitudes.csv from *input_dir* and return a list of
    (band_name, low_hz, high_hz, sum_magnitude_db) tuples.

    Looks for the CSV at:
      1. <input_dir>/analysis/band_magnitudes.csv
      2. experiments/<input_dir>/analysis/band_magnitudes.csv

    Returns None (with a warning) if the file is not found, so callers can
    skip the band plot gracefully when the CSV has not yet been generated.
    """
    candidates = [
        os.path.join(input_dir, "analysis", "band_magnitudes.csv"),
        os.path.join("experiments", input_dir, "analysis", "band_magnitudes.csv"),
    ]

    csv_path = None
    for cand in candidates:
        if os.path.isfile(cand):
            csv_path = cand
            break

    if not csv_path:
        print(f"  Warning: band_magnitudes.csv not found for '{input_dir}' — skipping band plot.", file=sys.stderr)
        return None

    bands = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sum_db = float(row["sum_magnitude_db"]) if row["sum_magnitude_db"] not in ("nan", "") else float("nan")
            bands.append((
                row["band_name"],
                float(row["low_hz"]),
                float(row["high_hz"]),
                sum_db,
            ))
    return bands


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _configure_log_xaxis(ax):
    """Apply consistent log-scale x-axis tick formatting to *ax*."""
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_resonance_response(experiments, output_dir):
    """
    Plot transfer-function magnitude (dB) and phase (deg) for all experiments.

    Parameters
    ----------
    experiments : list of (name, freqs, magnitudes, phases, piezo_dbfs, mic_dbfs)
    output_dir  : str  – destination directory (must already exist)
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    title = "Resonance Response"
    if len(experiments) == 1:
        title += f" — {experiments[0][0]}"
    fig.suptitle(title)

    for idx, (name, freqs, mags, phases, _, _) in enumerate(experiments):
        color = colors[idx % len(colors)]
        lw = 2 if idx == 0 else 1.5
        alpha = 1.0 if idx == 0 else 0.9
        ax1.semilogx(freqs, 20 * np.log10(mags),
                     label=name, linewidth=lw, alpha=alpha, color=color)
        ax2.semilogx(freqs, np.degrees(phases),
                     label=name, linewidth=lw, alpha=alpha, color=color)

    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both")
    ax1.legend(loc="best")

    ax2.set_ylabel("Phase (deg)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both")

    for ax in (ax1, ax2):
        _configure_log_xaxis(ax)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "resonance_response.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_rms_response(experiments, output_dir):
    """
    Plot RMS signal level (dBFS) vs frequency for all experiments.

    Parameters
    ----------
    experiments : list of (name, freqs, magnitudes, phases, piezo_dbfs, mic_dbfs)
    output_dir  : str
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])

    fig, ax = plt.subplots()
    title = "RMS Signal Level vs Frequency"
    if len(experiments) == 1:
        title += f" — {experiments[0][0]}"
    ax.set_title(title)

    for idx, (name, freqs, _, _, piezo_dbfs, mic_dbfs) in enumerate(experiments):
        color = colors[idx % len(colors)]
        lw = 2 if idx == 0 else 1.5
        alpha = 1.0 if idx == 0 else 0.9
        ax.semilogx(freqs, piezo_dbfs,
                    label=f"{name} piezo", linewidth=lw, alpha=alpha, color=color)
        ax.semilogx(freqs, mic_dbfs,
                    label=f"{name} mic", linewidth=lw, alpha=alpha,
                    linestyle="--", color=color)

    ax.set_ylabel("RMS Level (dBFS)")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both")
    ax.legend(loc="best")
    _configure_log_xaxis(ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "rms_response.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_sorted_magnitude_response(experiments, output_dir):
    """
    Plot magnitude values sorted in descending order for all experiments.

    Parameters
    ----------
    experiments : list of (name, freqs, magnitudes, phases, piezo_dbfs, mic_dbfs)
    output_dir  : str
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])

    fig, ax = plt.subplots()
    title = "Sorted Magnitude Response"
    if len(experiments) == 1:
        title += f" — {experiments[0][0]}"
    ax.set_title(title)

    for idx, (name, _, mags, _, _, _) in enumerate(experiments):
        color = colors[idx % len(colors)]
        lw = 2 if idx == 0 else 1.5
        alpha = 1.0 if idx == 0 else 0.9
        valid = np.isfinite(mags)
        sorted_mag = np.sort(np.asarray(mags[valid], dtype=float))[::-1]
        x = np.arange(len(sorted_mag))
        ax.plot(x, 20 * np.log10(sorted_mag),
                label=name, linewidth=lw, alpha=alpha, color=color)

    ax.set_xlabel("Magnitude rank")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both")
    ax.legend(loc="best")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "sorted_magnitude_response.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_band_magnitudes(band_experiments, output_dir):
    """
    Scatter plot of per-band summed magnitude (dB) for all experiments.

    Parameters
    ----------
    band_experiments : list of (name, bands)
        where bands is a list of (band_name, low_hz, high_hz, sum_db) tuples
        as returned by load_band_magnitudes_csv.
    output_dir : str
    """
    # Build a canonical band order from the first experiment that has data.
    band_order = None
    for _, bands in band_experiments:
        if bands is not None:
            band_order = [(b[0], b[1], b[2]) for b in bands]
            break

    if band_order is None:
        print("  Warning: no band magnitude data available — skipping band plot.", file=sys.stderr)
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])

    x_positions = np.arange(len(band_order))
    x_labels = [f"{name}\n{low:g}–{high:g} Hz" for name, low, high in band_order]

    # Spread multiple experiments slightly so overlapping points stay visible.
    n = len(band_experiments)
    offsets = np.linspace(-0.15, 0.15, n) if n > 1 else [0.0]

    fig, ax = plt.subplots(figsize=(max(8, len(band_order) * 1.6), 5))
    title = "Band Magnitudes"
    if len(band_experiments) == 1:
        title += f" — {band_experiments[0][0]}"
    ax.set_title(title)

    for idx, (name, bands) in enumerate(band_experiments):
        if bands is None:
            continue
        color = colors[idx % len(colors)]
        alpha = 1.0 if idx == 0 else 0.85

        # Build a lookup so bands missing from this experiment plot as NaN.
        band_lookup = {b[0]: b[3] for b in bands}
        y_values = [band_lookup.get(b[0], float("nan")) for b in band_order]

        xs = x_positions + offsets[idx]
        ys = np.array(y_values, dtype=float)

        # Plot finite points as filled circles, NaN points as an 'x' marker.
        finite = np.isfinite(ys)
        ax.scatter(xs[finite], ys[finite],
                   label=name, color=color, alpha=alpha, s=80, zorder=3)
        if np.any(~finite):
            ax.scatter(xs[~finite], np.zeros(np.sum(~finite)),
                       marker="x", color=color, alpha=alpha, s=60,
                       label=f"{name} (no data)", zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel("Sum magnitude (dB)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "band_magnitudes.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load frequency-sweep analysis CSVs from one or more experiment "
            "directories and generate combined overlay plots."
        )
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dirs",
        action="append",
        required=True,
        metavar="DIR",
        help=(
            "Path to an experiment directory containing analysis/report.csv "
            "(or report.csv). May be specified multiple times to overlay "
            "several experiments on the same plots."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where the PNG plots will be written. Created if absent.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load all experiments
    # ------------------------------------------------------------------
    experiments = []
    band_experiments = []
    for input_dir in args.input_dirs:
        name = os.path.basename(os.path.normpath(input_dir))
        print(f"Loading '{name}' from {input_dir} ...")
        try:
            freqs, mags, phases, piezo_dbfs, mic_dbfs = load_experiment_csv(input_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        experiments.append((name, freqs, mags, phases, piezo_dbfs, mic_dbfs))
        band_experiments.append((name, load_band_magnitudes_csv(input_dir)))

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join("experiments", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    plot_resonance_response(experiments, output_dir)
    plot_rms_response(experiments, output_dir)
    plot_sorted_magnitude_response(experiments, output_dir)
    plot_band_magnitudes(band_experiments, output_dir)

    print(f"\nAll plots written to: {output_dir}")


if __name__ == "__main__":
    main()
