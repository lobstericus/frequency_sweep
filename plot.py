"""
Standalone plotting tool for frequency sweep experiments.

Loads analysis data from one or more experiment directories (each containing
an analysis/report.csv file produced by analyze.py) and generates combined
overlay plots saved to --output-dir:

  1. resonance_response.png   - transfer-function magnitude & phase
  2. rms_response.png         - RMS signal level vs frequency
  3. sorted_magnitude_response.png - sorted magnitude response

Channel names and the list of resonance (source/sink) ratios plotted are
read from each experiment's parameters.json (as written by
sweep_and_capture.py), rather than being hard-coded, so the same code works
regardless of how many channels/resonances an experiment defines.

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

plt.rcParams["savefig.dpi"] = 300

from analyze import load_parameters, resonance_label

LINESTYLES = ["-", "--", ":", "-."]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def resolve_experiment_dir(input_dir):
    """Finds the directory that directly contains parameters.json for an
    experiment given either its full path or its experiments/<name> name."""
    candidates = [input_dir, os.path.join("experiments", input_dir)]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "parameters.json")):
            return cand
    raise FileNotFoundError(
        f"parameters.json not found for '{input_dir}'. Tried:\n" +
        "\n".join(f"  {os.path.join(c, 'parameters.json')}" for c in candidates)
    )


def load_experiment(input_dir):
    """
    Loads an experiment's parameters.json (for channel names and resonance
    definitions) and its report.csv, and returns a dict:

      {
        "name": str,
        "freqs": np.ndarray,
        "channel_names": [str, ...],
        "channels": {channel_name: dbfs_array, ...},
        "resonance_labels": [str, ...],
        "resonances": {label: {"magnitude": array, "phase": array}, ...},
      }

    Looks for report.csv at:
      1. <experiment_dir>/analysis/report.csv
      2. <experiment_dir>/report.csv
    """
    experiment_dir = resolve_experiment_dir(input_dir)
    params = load_parameters(experiment_dir)
    channels_cfg = sorted(params["channels"], key=lambda ch: ch.get("id", 0))
    channel_names = [ch["name"] for ch in channels_cfg]
    resonances_cfg = params.get("compute_resonance", [])
    resonance_labels = [resonance_label(r) for r in resonances_cfg]

    candidates = [
        os.path.join(experiment_dir, "analysis", "report.csv"),
        os.path.join(experiment_dir, "report.csv"),
    ]
    report_path = next((c for c in candidates if os.path.isfile(c)), None)
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

    channels = {}
    for name in channel_names:
        rms = np.array([float(row[f"{name}_rms"]) for row in rows], dtype=float)
        channels[name] = np.array([20 * np.log10(v) if v > 0 else -np.inf for v in rms])

    resonances = {}
    for r, label in zip(resonances_cfg, resonance_labels):
        resonances[label] = {
            "source": r["source"],
            "sink": r["sink"],
            "magnitude": np.array([float(row[f"{label}_magnitude"]) for row in rows], dtype=float),
            "phase": np.array([float(row[f"{label}_phase_radians"]) for row in rows], dtype=float),
        }

    name = os.path.basename(os.path.normpath(input_dir))
    return {
        "name": name,
        "freqs": freqs,
        "channel_names": channel_names,
        "channels": channels,
        "resonance_labels": resonance_labels,
        "resonances": resonances,
    }


def load_band_magnitudes(input_dir, resonance_labels):
    """
    Load band_magnitudes.csv for an experiment and return a dict mapping
    resonance_label -> list of (band_name, low_hz, high_hz, sum_magnitude_db)
    tuples.

    Looks for the CSV at:
      1. <experiment_dir>/analysis/band_magnitudes.csv
      2. <experiment_dir>/band_magnitudes.csv

    Returns None (with a warning) if the file is not found, so callers can
    skip the band plot gracefully when the CSV has not yet been generated.
    """
    try:
        experiment_dir = resolve_experiment_dir(input_dir)
    except FileNotFoundError as exc:
        print(f"  Warning: {exc} — skipping band plot.", file=sys.stderr)
        return None

    candidates = [
        os.path.join(experiment_dir, "analysis", "band_magnitudes.csv"),
        os.path.join(experiment_dir, "band_magnitudes.csv"),
    ]
    csv_path = next((c for c in candidates if os.path.isfile(c)), None)
    if not csv_path:
        print(f"  Warning: band_magnitudes.csv not found for '{input_dir}' — skipping band plot.", file=sys.stderr)
        return None

    bands_by_resonance = {label: [] for label in resonance_labels}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sum_db = float(row["sum_magnitude_db"]) if row["sum_magnitude_db"] not in ("nan", "") else float("nan")
            bands_by_resonance.setdefault(row["resonance_name"], []).append((
                row["band_name"],
                float(row["low_hz"]),
                float(row["high_hz"]),
                sum_db,
            ))
    return bands_by_resonance


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
    Plot transfer-function magnitude (dB) and phase (deg) vs frequency,
    comparing experiments. Writes one PNG per configured resonance
    (source/sink pair), so each plot only ever compares two signals.

    Parameters
    ----------
    experiments : list of dicts, as returned by load_experiment()
    output_dir  : str  – destination directory (must already exist)
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])

    labels_seen = []
    for exp in experiments:
        for label in exp["resonance_labels"]:
            if label not in labels_seen:
                labels_seen.append(label)

    for label in labels_seen:
        relevant = [exp for exp in experiments if label in exp["resonances"]]
        if not relevant:
            continue
        source = relevant[0]["resonances"][label]["source"]
        sink = relevant[0]["resonances"][label]["sink"]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        title = f"Resonance Response ({source} → {sink})"
        if len(relevant) == 1:
            title += f" — {relevant[0]['name']}"
        fig.suptitle(title)

        for idx, exp in enumerate(relevant):
            color = colors[idx % len(colors)]
            lw = 2 if idx == 0 else 1.5
            alpha = 1.0 if idx == 0 else 0.9
            mags = exp["resonances"][label]["magnitude"]
            phases = exp["resonances"][label]["phase"]
            ax1.semilogx(exp["freqs"], 20 * np.log10(mags),
                         label=exp["name"], linewidth=lw, alpha=alpha, color=color)
            ax2.semilogx(exp["freqs"], np.degrees(phases),
                         label=exp["name"], linewidth=lw, alpha=alpha, color=color)

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
        channel_suffix = f"{source}_{sink}".replace(" ", "_")
        out_path = os.path.join(output_dir, f"resonance_response_{channel_suffix}.png")
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_rms_response(experiments, output_dir):
    """
    Plot RMS signal level (dBFS) vs frequency for every channel of every
    experiment.

    Parameters
    ----------
    experiments : list of dicts, as returned by load_experiment()
    output_dir  : str
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])

    fig, ax = plt.subplots()
    title = "RMS Signal Level vs Frequency"
    if len(experiments) == 1:
        title += f" — {experiments[0]['name']}"
    ax.set_title(title)

    for idx, exp in enumerate(experiments):
        color = colors[idx % len(colors)]
        lw = 2 if idx == 0 else 1.5
        alpha = 1.0 if idx == 0 else 0.9
        for c_idx, channel_name in enumerate(exp["channel_names"]):
            ls = LINESTYLES[c_idx % len(LINESTYLES)]
            ax.semilogx(exp["freqs"], exp["channels"][channel_name],
                        label=f"{exp['name']} {channel_name}", linewidth=lw, alpha=alpha,
                        linestyle=ls, color=color)

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
    Plot magnitude values sorted in descending order, comparing experiments.
    Writes one PNG per configured resonance (source/sink pair), so each plot
    only ever compares two signals.

    Parameters
    ----------
    experiments : list of dicts, as returned by load_experiment()
    output_dir  : str
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])

    labels_seen = []
    for exp in experiments:
        for label in exp["resonance_labels"]:
            if label not in labels_seen:
                labels_seen.append(label)

    for label in labels_seen:
        relevant = [exp for exp in experiments if label in exp["resonances"]]
        if not relevant:
            continue
        source = relevant[0]["resonances"][label]["source"]
        sink = relevant[0]["resonances"][label]["sink"]

        fig, ax = plt.subplots()
        title = f"Sorted Magnitude Response ({source} → {sink})"
        if len(relevant) == 1:
            title += f" — {relevant[0]['name']}"
        ax.set_title(title)

        for idx, exp in enumerate(relevant):
            color = colors[idx % len(colors)]
            lw = 2 if idx == 0 else 1.5
            alpha = 1.0 if idx == 0 else 0.9
            mags = exp["resonances"][label]["magnitude"]
            valid = np.isfinite(mags)
            sorted_mag = np.sort(np.asarray(mags[valid], dtype=float))[::-1]
            x = np.arange(len(sorted_mag))
            ax.plot(x, 20 * np.log10(sorted_mag),
                    label=exp["name"], linewidth=lw, alpha=alpha, color=color)

        ax.set_xlabel("Magnitude rank")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, which="both")
        ax.legend(loc="best")

        plt.tight_layout()
        channel_suffix = f"{source}_{sink}".replace(" ", "_")
        out_path = os.path.join(output_dir, f"sorted_magnitude_response_{channel_suffix}.png")
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_band_magnitudes(band_experiments, output_dir):
    """
    Scatter plot of per-band summed magnitude (dB) for every resonance of
    every experiment.

    Parameters
    ----------
    band_experiments : list of (name, bands_by_resonance)
        where bands_by_resonance is a dict mapping resonance_label -> list of
        (band_name, low_hz, high_hz, sum_db) tuples, as returned by
        load_band_magnitudes().
    output_dir : str
    """
    # Build a canonical band order from the first experiment/resonance that has data.
    band_order = None
    for _, bands_by_resonance in band_experiments:
        if bands_by_resonance:
            for bands in bands_by_resonance.values():
                if bands:
                    band_order = [(b[0], b[1], b[2]) for b in bands]
                    break
        if band_order is not None:
            break

    if band_order is None:
        print("  Warning: no band magnitude data available — skipping band plot.", file=sys.stderr)
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])
    multi_resonance = len({
        label for _, bands_by_resonance in band_experiments
        for label in (bands_by_resonance or {})
    }) > 1

    x_positions = np.arange(len(band_order))
    x_labels = [f"{name}\n{low:g}–{high:g} Hz" for name, low, high in band_order]

    fig, ax = plt.subplots(figsize=(max(8, len(band_order) * 1.6), 5))
    title = "Band Magnitudes"
    if len(band_experiments) == 1:
        title += f" — {band_experiments[0][0]}"
    ax.set_title(title)

    for idx, (name, bands_by_resonance) in enumerate(band_experiments):
        if not bands_by_resonance:
            continue
        color = colors[idx % len(colors)]
        alpha = 1.0 if idx == 0 else 0.85

        for r_idx, (label, bands) in enumerate(bands_by_resonance.items()):
            if not bands:
                continue
            ls = LINESTYLES[r_idx % len(LINESTYLES)]
            legend_label = f"{name} [{label}]" if multi_resonance else name

            # Build a lookup so bands missing from this experiment plot as NaN.
            band_lookup = {b[0]: b[3] for b in bands}
            y_values = [band_lookup.get(b[0], float("nan")) for b in band_order]

            xs = x_positions
            ys = np.array(y_values, dtype=float)

            # Plot finite points as filled circles, NaN points as an 'x' marker.
            finite = np.isfinite(ys)
            ax.plot(xs[finite], ys[finite],
                       label=legend_label, color=color, alpha=alpha, linestyle=ls, zorder=3)
            if np.any(~finite):
                ax.scatter(xs[~finite], np.zeros(np.sum(~finite)),
                           marker="x", color=color, alpha=alpha, s=60,
                           label=f"{legend_label} (no data)", zorder=3)

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
        default=None,
        metavar="DIR",
        help="Directory where the PNG plots will be written. Created if absent. "
             "Defaults to the first --input-dir if omitted.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dirs[0]

    # ------------------------------------------------------------------
    # Load all experiments
    # ------------------------------------------------------------------
    experiments = []
    band_experiments = []
    for input_dir in args.input_dirs:
        name = os.path.basename(os.path.normpath(input_dir))
        print(f"Loading '{name}' from {input_dir} ...")
        try:
            exp = load_experiment(input_dir)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        experiments.append(exp)
        band_experiments.append((name, load_band_magnitudes(input_dir, exp["resonance_labels"])))

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join("experiments", output_dir)
    output_dir = os.path.join(output_dir, "plots")
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
