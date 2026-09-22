"""
Stepped-sine resonance analysis.

Processes recorded audio files (.wav) from experiments/<input-dir>/samples/,
extracts the test frequency from the sample filename (sample_xxxxxx.x.wav),
computes transfer function magnitude and phase using a single-frequency DFT,
calculates RMS levels, and outputs to experiments/<input-dir>/analysis/:
  1. report.csv           - frequency, per-channel RMS, per-resonance magnitude/phase
  2. average_magnitude.csv - mean transfer-function magnitude (dB) summary

Channel names (assigned to each .wav channel, in order) and the list of
source/sink pairs used to compute resonance ratios are read from
experiments/<input-dir>/parameters.json, as written by sweep_and_capture.py.
"""

import argparse
import csv
import json
import os
import sys
import wave
import numpy as np


resonance_bands = {
    "SUB": (0, 270),

    "A0": (270, 280),
    "A1": (460, 470),
    "B1-": (440, 470),
    "B1+": (470, 540),

    "X0": (540, 800),
    "X4": (800, 1120),
    
    "X8": (1120, 1520),
    "X13": (1520, 2000),

    "Bridge hill": (2000, 3000),

    "y0": (3000, 4500),

    "Secondary/nasal hill": (4500, 6000),
}

# ---------------------------------------------------------------------


def load_parameters(input_dir):
    """Loads experiments/<input_dir>/parameters.json (channel names/order and
    the source/sink pairs used to compute resonance ratios)."""
    params_path = os.path.join(input_dir, "parameters.json")
    with open(params_path) as f:
        return json.load(f)


def resonance_label(resonance):
    """Returns the display/column name for a {source, sink} resonance entry,
    using an explicit "name" if given, otherwise "<sink>_over_<source>"."""
    return resonance.get("name") or f"{resonance['sink']}_over_{resonance['source']}"


def bin_ratio(source, sink, freq, fs):
    """
    Removes DC bias, applies a Hann window, computes the single-frequency
    discrete Fourier transform (DFT) at the exact test frequency `freq`,
    and returns the complex transfer function ratio sink/source (magnitude, phase).
    """
    source = source - np.mean(source)
    sink = sink - np.mean(sink)

    window = np.hanning(len(source))
    source_w = source * window
    sink_w = sink * window

    t = np.arange(len(source_w)) / fs
    basis = np.exp(-2j * np.pi * freq * t)

    source_val = np.dot(source_w, basis)
    sink_val = np.dot(sink_w, basis)

    if np.abs(source_val) < 1e-12:
        return np.nan, np.nan

    H = sink_val / source_val
    return np.abs(H), np.angle(H)


def load_channel_wav(wav_path, channel_names):
    """
    Loads a multi-channel PCM WAV file and returns a dict mapping each name in
    channel_names (in .wav channel order) to its float32 waveform, plus fs.
    """
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if len(channel_names) != n_channels:
        raise ValueError(
            f"parameters.json lists {len(channel_names)} channel(s) but "
            f"{wav_path} has {n_channels} channel(s)"
        )

    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483647.0
    elif sampwidth == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes in {wav_path}")

    data = data.reshape(-1, n_channels)
    return {name: data[:, i] for i, name in enumerate(channel_names)}, fs


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


def analyze_samples(input_dir, analysis_dir=None):
    """
    Process recorded audio samples from input_dir/samples, using channel
    names and resonance source/sink pairs from input_dir/parameters.json.
    Computes DFT magnitude/phase per resonance and RMS levels per channel,
    writes report.csv in analysis_dir, and returns
    (freqs, channel_names, channel_rms_dbfs, resonances, resonance_mags, resonance_phases).

    channel_rms_dbfs, resonance_mags, and resonance_phases are dicts keyed by
    channel name / resonance label, each holding an array parallel to freqs.
    """
    if analysis_dir is None:
        analysis_dir = os.path.join(input_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    settings = load_parameters(input_dir)
    channels_cfg = sorted(settings["channels"], key=lambda ch: ch.get("id", 0))
    channel_names = [ch["name"] for ch in channels_cfg]
    resonances = settings.get("compute_resonance", [])
    resonance_labels = [resonance_label(r) for r in resonances]

    samples_dir = os.path.join(input_dir, "samples")
    sample_files = get_sample_files(samples_dir)

    n_samples = len(sample_files)
    freqs = np.array([f for f, _ in sample_files], dtype=float)
    channel_rms_dbfs = {name: np.zeros(n_samples) for name in channel_names}
    resonance_mags = {label: np.zeros(n_samples) for label in resonance_labels}
    resonance_phases = {label: np.zeros(n_samples) for label in resonance_labels}

    report_path = os.path.join(analysis_dir, "report.csv")
    with open(report_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["frequency_hz"]
        header += [f"{name}_rms" for name in channel_names]
        for label in resonance_labels:
            header += [f"{label}_magnitude", f"{label}_phase_radians", f"{label}_phase_degrees"]
        writer.writerow(header)

        for i, (f, wav_path) in enumerate(sample_files):
            channels, fs = load_channel_wav(wav_path, channel_names)

            row = [f]
            rms_parts = []
            for name in channel_names:
                rms = float(np.sqrt(np.mean(channels[name] ** 2)))
                dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf
                channel_rms_dbfs[name][i] = dbfs
                row.append(rms)
                rms_parts.append(f"{name}_rms = {dbfs:5.2f} dBFS")

            resonance_parts = []
            for r, label in zip(resonances, resonance_labels):
                mag, ph = bin_ratio(channels[r["source"]], channels[r["sink"]], f, fs)
                resonance_mags[label][i] = mag
                resonance_phases[label][i] = ph
                row += [mag, ph, np.degrees(ph)]
                resonance_parts.append(f"{label} |H| = {mag:5.2f} phase = {np.degrees(ph):7.2f} deg")

            writer.writerow(row)
            print(f"{f:7.2f} Hz  " + "  ".join(resonance_parts + rms_parts))

    return freqs, channel_names, channel_rms_dbfs, resonance_labels, resonance_mags, resonance_phases


def compute_average_magnitude_db(freqs, magnitudes):
    """Compute the mean transfer-function magnitude in dB across all valid frequency points."""
    valid = np.isfinite(freqs) & np.isfinite(magnitudes) & (magnitudes > 0)
    if not np.any(valid):
        return np.nan
    return float(np.mean(20 * np.log10(magnitudes[valid])))


def save_average_magnitude_csv(output_dir, experiment_name, resonance_averages):
    """Write mean magnitude (dB) for each resonance to average_magnitude.csv.

    resonance_averages: dict mapping resonance_label -> average_magnitude_db.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "average_magnitude.csv")

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["experiment_name", "resonance_name", "average_magnitude_db"])
        for resonance_name, avg in resonance_averages.items():
            writer.writerow([experiment_name, resonance_name, avg])

    return output_path


def compute_band_magnitudes(freqs, magnitudes, bands):
    """
    Sum the linear transfer-function magnitudes that fall within each named
    frequency band, then convert the total to dB.

    Parameters
    ----------
    freqs      : array-like of measured frequencies (Hz)
    magnitudes : array-like of linear |H| values (same length as freqs)
    bands      : dict mapping band_name -> (low_hz, high_hz)

    Returns
    -------
    dict mapping band_name -> sum_db  (float, or nan if no points fall in band)
    """
    freqs = np.asarray(freqs, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)
    valid = np.isfinite(freqs) & np.isfinite(magnitudes) & (magnitudes > 0)

    results = {}
    for band_name, (low, high) in bands.items():
        in_band = valid & (freqs >= low) & (freqs < high)
        if not np.any(in_band):
            results[band_name] = float("nan")
        else:
            total_linear = float(np.mean(magnitudes[in_band]))
            results[band_name] = 20 * np.log10(total_linear)
    return results


def save_band_magnitudes_csv(output_dir, band_results_by_resonance, bands):
    """
    Write per-band, per-resonance summed magnitude (dB) to band_magnitudes.csv.

    Parameters
    ----------
    output_dir                : destination directory
    band_results_by_resonance : dict mapping resonance_label -> band_results
                                 (each from compute_band_magnitudes)
    bands                     : original resonance_bands dict (used to record Hz ranges)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "band_magnitudes.csv")

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["resonance_name", "band_name", "low_hz", "high_hz", "sum_magnitude_db"])
        for resonance_name, band_results in band_results_by_resonance.items():
            for band_name, sum_db in band_results.items():
                low, high = bands[band_name]
                writer.writerow([resonance_name, band_name, low, high, sum_db])

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze recorded frequency sweep audio samples and write report CSVs."
    )
    parser.add_argument(
        "--input-dir", "--output-dir", default="default",
        help="Name or path of the experiment directory (e.g. experiments/<input-dir>).",
    )
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

    analysis_dir = os.path.join(input_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    samples_dir = os.path.join(input_dir, "samples")
    if not os.path.isdir(samples_dir) or not any(
        f.startswith("sample_") and f.endswith(".wav") for f in os.listdir(samples_dir)
    ):
        print(f"Error: No sample WAV files found in {samples_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing recorded samples from {samples_dir}...")
    (freqs, channel_names, channel_rms_dbfs,
     resonance_labels, resonance_mags, resonance_phases) = analyze_samples(input_dir, analysis_dir)

    resonance_averages = {
        label: compute_average_magnitude_db(freqs, resonance_mags[label])
        for label in resonance_labels
    }
    average_magnitude_csv = save_average_magnitude_csv(analysis_dir, experiment_name, resonance_averages)

    print(f"\nAverage magnitude:")
    for label, avg in resonance_averages.items():
        print(f"  {experiment_name} [{label}]: {avg:.6f} dB")
    print(f"Saved average magnitude to: {average_magnitude_csv}")

    print(f"\nComputing average band magnitudes for {len(resonance_bands)} bands.")
    for band_name, (low, high) in resonance_bands.items():
        print(f"  '{band_name}' with range {low}-{high} Hz")

    band_results_by_resonance = {
        label: compute_band_magnitudes(freqs, resonance_mags[label], resonance_bands)
        for label in resonance_labels
    }
    band_csv = save_band_magnitudes_csv(analysis_dir, band_results_by_resonance, resonance_bands)

    print(f"\nBand magnitudes (sum of |H| in band, dB):")
    for label, band_results in band_results_by_resonance.items():
        print(f"  [{label}]")
        for band_name, sum_db in band_results.items():
            low, high = resonance_bands[band_name]
            db_str = f"{sum_db:.4f} dB" if not np.isnan(sum_db) else "no data"
            print(f"    {band_name:30s} [{low:5g}–{high:5g} Hz]: {db_str}")
    print(f"Saved band magnitudes to: {band_csv}")

    print(f"\nRun plot.py to generate plots from {analysis_dir}/report.csv")
