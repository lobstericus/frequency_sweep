"""
Stepped-sine resonance analysis.

Processes recorded audio files (.wav) from experiments/<input-dir>/samples/,
extracts the test frequency from the sample filename (sample_xxxxxx.x.wav),
computes transfer function magnitude and phase using FFT, calculates RMS
levels, and outputs:
  1. report.csv - frequency, magnitude, phase, piezo_rms, mic_rms
  2. resonance_response.png - transfer-function magnitude/phase plots
  3. rms_response.png - RMS level vs frequency plots
  4. sorted_magnitude_response.png - sorted magnitude response plot
  5. response_area.csv - response area (dB-Hz) summary
"""

import argparse
import csv
import os
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter


# ---------------------------------------------------------------------
# Config & baseline
# ---------------------------------------------------------------------
RESPONSE_AREA_BASELINE_DB = -40


def bin_ratio(piezo, mic, freq, fs):
    """
    Removes DC bias, applies a Hann window, computes rfft of both
    signals, and returns the complex ratio mic/piezo at the bin
    nearest to `freq`.
    """
    piezo = piezo - np.mean(piezo)
    mic = mic - np.mean(mic)

    window = np.hanning(len(piezo))
    piezo_w = piezo * window
    mic_w = mic * window

    freqs = np.fft.rfftfreq(len(piezo_w), d=1 / fs)
    piezo_fft = np.fft.rfft(piezo_w)
    mic_fft = np.fft.rfft(mic_w)

    bin_idx = np.argmin(np.abs(freqs - freq))

    if np.abs(piezo_fft[bin_idx]) < 1e-12:
        return np.nan, np.nan

    H = mic_fft[bin_idx] / piezo_fft[bin_idx]
    return np.abs(H), np.angle(H)


def load_channel_wav(wav_path):
    """
    Loads a 2-channel PCM WAV file and returns (piezo, mic, fs) as float32 arrays.
    Channel 1 is piezo, Channel 2 is mic.
    """
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483647.0
    elif sampwidth == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes in {wav_path}")

    if n_channels == 2:
        data = data.reshape(-1, 2)
        piezo = data[:, 0]
        mic = data[:, 1]
    elif n_channels == 1:
        piezo = data
        mic = data
    else:
        data = data.reshape(-1, n_channels)
        piezo = data[:, 0]
        mic = data[:, 1]

    return piezo, mic, fs


def get_sample_files(samples_dir):
    """
    Finds all sample_*.wav files in samples_dir, extracts frequencies from
    the filename (splitting off 'sample_' prefix and '.wav' extension), and
    returns a sorted list of (freq, filepath) tuples.
    """
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    samples = []
    for fname in os.listdir(samples_dir):
        if fname.startswith("sample_") and fname.endswith(".wav"):
            freq_str = fname[len("sample_"):-len(".wav")]
            try:
                freq = float(freq_str)
                samples.append((freq, os.path.join(samples_dir, fname)))
            except ValueError:
                continue

    if not samples:
        raise FileNotFoundError(f"No valid sample_*.wav files found in {samples_dir}")

    samples.sort(key=lambda item: item[0])
    return samples


def analyze_samples(input_dir):
    """
    Process recorded audio samples from input_dir/samples.
    Computes FFT magnitude/phase, RMS levels, writes report.csv,
    and returns (freqs, magnitudes, phases, piezo_dbfs_arr, mic_dbfs_arr).
    """
    samples_dir = os.path.join(input_dir, "samples")
    sample_files = get_sample_files(samples_dir)

    n_samples = len(sample_files)
    freqs = np.array([f for f, _ in sample_files], dtype=float)
    magnitudes = np.zeros(n_samples)
    phases = np.zeros(n_samples)
    piezo_dbfs_arr = np.zeros(n_samples)
    mic_dbfs_arr = np.zeros(n_samples)

    report_path = os.path.join(input_dir, "resonance_response.csv")
    with open(report_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frequency_hz",
            "magnitude",
            "phase_radians",
            "phase_degrees",
            "piezo_rms",
            "mic_rms"
        ])

        for i, (f, wav_path) in enumerate(sample_files):
            piezo, mic, fs = load_channel_wav(wav_path)
            mag, ph = bin_ratio(piezo, mic, f, fs)
            magnitudes[i] = mag
            phases[i] = ph
            piezo_rms = float(np.sqrt(np.mean(piezo ** 2)))
            mic_rms = float(np.sqrt(np.mean(mic ** 2)))
            piezo_dbfs = 20 * np.log10(piezo_rms) if piezo_rms > 0 else -np.inf
            mic_dbfs = 20 * np.log10(mic_rms) if mic_rms > 0 else -np.inf
            piezo_dbfs_arr[i] = piezo_dbfs
            mic_dbfs_arr[i] = mic_dbfs

            writer.writerow([
                f,
                mag,
                ph,
                np.degrees(ph),
                piezo_rms,
                mic_rms,
            ])
            print(f"{f:7.2f} Hz  |H| = {mag:5.2f}  phase = {np.degrees(ph):5.2f} deg  "
                  f"piezo_rms = {piezo_dbfs:5.2f} dBFS  mic_rms = {mic_dbfs:5.2f} dBFS")

    return freqs, magnitudes, phases, piezo_dbfs_arr, mic_dbfs_arr


def load_experiment_csv(experiment_name):
    """Load a saved experiment report CSV and return arrays for plotting."""
    if os.path.isfile(os.path.join(experiment_name, "report.csv")):
        report_path = os.path.join(experiment_name, "report.csv")
    elif os.path.isfile(os.path.join("experiments", experiment_name, "report.csv")):
        report_path = os.path.join("experiments", experiment_name, "report.csv")
    elif os.path.isfile(experiment_name) and experiment_name.endswith(".csv"):
        report_path = experiment_name
    else:
        report_path = os.path.join("experiments", experiment_name, "report.csv")

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Experiment report not found: {report_path}")

    with open(report_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Experiment report is empty: {report_path}")

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


def compute_response_area(freqs, magnitudes):
    """Compute dB-Hz response area above RESPONSE_AREA_BASELINE_DB."""
    valid = np.isfinite(freqs) & np.isfinite(magnitudes)
    if not np.any(valid):
        return np.nan

    freqs = np.asarray(freqs[valid], dtype=float)
    magnitudes = np.asarray(magnitudes[valid], dtype=float)
    order = np.argsort(freqs)
    magnitude_db = 20 * np.log10(magnitudes[order])
    area_height_db = np.maximum(magnitude_db - RESPONSE_AREA_BASELINE_DB, 0)
    return float(np.trapezoid(area_height_db, freqs[order]))


def save_response_area_csv(output_dir, experiment_areas):
    """Write area-under-curve values for each experiment to response_area.csv."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "response_area.csv")

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["experiment_name", "response_area_db_hz"])
        for experiment_name, area in experiment_areas.items():
            writer.writerow([experiment_name, area])

    return output_path


def plot_response_curves(freqs, magnitudes, phases, piezo_dbfs_arr, mic_dbfs_arr,
                         test_name="default", output_dir=None, overlay=None, show=True):
    """Plot the swept transfer-function magnitude/phase and RMS traces."""
    if output_dir is None:
        output_dir = os.path.join("experiments", test_name)
    os.makedirs(output_dir, exist_ok=True)
    if overlay is None:
        overlay = []

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [
        "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"
    ])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(f"Resonance Response — {test_name}")
    ax1.semilogx(freqs, 20 * np.log10(magnitudes), label=test_name, linewidth=2, color=colors[0])
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both")

    ax2.semilogx(freqs, np.degrees(phases), label=test_name, linewidth=2, color=colors[0])
    ax2.set_ylabel("Phase (deg)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both")

    for idx, (overlay_name, overlay_freqs, overlay_magnitudes, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs) in enumerate(overlay, start=1):
        color = colors[idx % len(colors)]
        ax1.semilogx(overlay_freqs, 20 * np.log10(overlay_magnitudes), label=overlay_name, alpha=0.9, color=color)
        ax2.semilogx(overlay_freqs, np.degrees(overlay_phases), label=overlay_name, alpha=0.9, color=color)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    ax1.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resonance_response.png"))

    fig_rms, ax_rms = plt.subplots()
    ax_rms.semilogx(freqs, piezo_dbfs_arr, label=f"{test_name} piezo", linewidth=2, color=colors[0])
    ax_rms.semilogx(freqs, mic_dbfs_arr, label=f"{test_name} mic", linewidth=2, linestyle="--", color=colors[0])
    ax_rms.set_title(f"RMS Signal Level vs Frequency — {test_name}")
    ax_rms.set_ylabel("RMS Level (dBFS)")
    ax_rms.set_xlabel("Frequency (Hz)")
    ax_rms.grid(True, which="both")

    for idx, (overlay_name, overlay_freqs, overlay_magnitudes, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs) in enumerate(overlay, start=1):
        color = colors[idx % len(colors)]
        ax_rms.semilogx(overlay_freqs, overlay_piezo_dbfs, label=f"{overlay_name} piezo", alpha=0.9, color=color)
        ax_rms.semilogx(overlay_freqs, overlay_mic_dbfs, label=f"{overlay_name} mic", alpha=0.9, linestyle="--", color=color)

    ax_rms.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
    ax_rms.xaxis.set_major_formatter(ScalarFormatter())
    ax_rms.xaxis.set_minor_formatter(NullFormatter())
    plt.setp(ax_rms.get_xticklabels(), rotation=45, ha="right")
    ax_rms.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rms_response.png"))

    fig_sorted, ax_sorted = plt.subplots()
    series = [(test_name, magnitudes)]
    series.extend([(name, mag_vals) for name, _, mag_vals, _, _, _ in overlay])

    for idx, (name, mag_vals) in enumerate(series):
        color = colors[idx % len(colors)]
        valid = np.isfinite(mag_vals)
        sorted_mag = np.sort(np.asarray(mag_vals[valid], dtype=float))[::-1]
        x = np.arange(len(sorted_mag))
        ax_sorted.plot(x, 20 * np.log10(sorted_mag), label=name, linewidth=2, color=color)

    ax_sorted.set_title(f"Sorted Magnitude Response — {test_name}")
    ax_sorted.set_xlabel("Magnitude rank")
    ax_sorted.set_ylabel("Magnitude (dB)")
    ax_sorted.grid(True, which="both")
    ax_sorted.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sorted_magnitude_response.png"))
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze recorded frequency sweep audio samples and generate reports and plots."
    )
    parser.add_argument("--input-dir", "--output-dir", default="default",
                        help="Name or path of the experiment directory (e.g. experiments/<input-dir>).")
    parser.add_argument("--plot", dest="plots", action="append", default=[],
                        metavar="EXPERIMENT_NAME",
                        help="Load and overlay a saved experiment from experiments/<EXPERIMENT_NAME>/report.csv. Can be passed multiple times.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display interactive matplotlib window.")
    args = parser.parse_args()

    input_arg = args.input_dir
    if os.path.isdir(input_arg):
        input_dir = input_arg
        experiment_name = os.path.basename(os.path.normpath(input_arg))
    elif os.path.isdir(os.path.join("experiments", input_arg)):
        input_dir = os.path.join("experiments", input_arg)
        experiment_name = input_arg
    else:
        input_dir = os.path.join("experiments", input_arg)
        experiment_name = input_arg

    samples_dir = os.path.join(input_dir, "samples")
    has_samples = os.path.isdir(samples_dir) and any(
        f.startswith("sample_") and f.endswith(".wav") for f in os.listdir(samples_dir)
    )

    if has_samples:
        print(f"Analyzing recorded samples from {samples_dir}...")
        freqs, mags, phases, piezo_dbfs_arr, mic_dbfs_arr = analyze_samples(input_dir)
        overlay = []
        experiment_areas = {experiment_name: compute_response_area(freqs, mags)}
    elif args.plots:
        source_experiment = args.plots[0]
        try:
            freqs, mags, phases, piezo_dbfs_arr, mic_dbfs_arr = load_experiment_csv(source_experiment)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        overlay = []
        experiment_areas = {source_experiment: compute_response_area(freqs, mags)}
        for exp_name in args.plots[1:]:
            try:
                overlay_freqs, overlay_mags, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs = load_experiment_csv(exp_name)
                overlay.append((exp_name, overlay_freqs, overlay_mags, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs))
                experiment_areas[exp_name] = compute_response_area(overlay_freqs, overlay_mags)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Error: {exc}")
                sys.exit(1)
    else:
        report_path = os.path.join(input_dir, "report.csv")
        if not os.path.exists(report_path):
            print(f"Error: No sample WAV files found in {samples_dir}, and experiment report not found: {report_path}")
            sys.exit(1)

        freqs, mags, phases, piezo_dbfs_arr, mic_dbfs_arr = load_experiment_csv(experiment_name)
        overlay = []
        experiment_areas = {experiment_name: compute_response_area(freqs, mags)}

    if has_samples and args.plots:
        for exp_name in args.plots:
            try:
                overlay_freqs, overlay_mags, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs = load_experiment_csv(exp_name)
                overlay.append((exp_name, overlay_freqs, overlay_mags, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs))
                experiment_areas[exp_name] = compute_response_area(overlay_freqs, overlay_mags)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Error: {exc}")
                sys.exit(1)

    plot_response_curves(freqs, mags, phases, piezo_dbfs_arr, mic_dbfs_arr,
                         test_name=experiment_name, output_dir=input_dir,
                         overlay=overlay, show=not args.no_show)

    response_area_csv = save_response_area_csv(input_dir, experiment_areas)
    print(f"\nResponse area above {RESPONSE_AREA_BASELINE_DB} dB:")
    for exp_name, area in experiment_areas.items():
        print(f"  {exp_name}: {area:.6f} dB-Hz")
    print(f"Saved area summary to: {response_area_csv}")