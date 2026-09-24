"""Analyze multi-channel stepped-sine resonance measurements."""

import argparse
import csv
import json
import os
import sys
import wave

import numpy as np


resonance_bands = {
    "A0": (270, 280), # Helmholtz air resonance of cavity
    "CBR": (400,420), # Center bout rombold - top plate waste resonance
    "A1": (460, 470), # Longitudinal air resonance 
    "B1-": (440, 470), # First bending mode. Mix of top and bottom
    "B1+": (470, 540), # Second bending mode. Mostly top plate
    "Bridge hill": (2000, 3000), # Bridge resonance. Sololist quality violin
}


def load_parameters(input_dir):
    """Load the capture-time channel and resonance configuration."""
    with open(os.path.join(input_dir, "parameters.json")) as parameter_file:
        return json.load(parameter_file)


def resonance_label(resonance):
    """Return the report-column name for a configured source/sink ratio."""
    return resonance.get("name") or f"{resonance['sink']}_over_{resonance['source']}"


def bin_ratio(source, sink, freq, fs):
    """Return the single-frequency DFT transfer-function ratio sink/source."""
    source = source - np.mean(source)
    sink = sink - np.mean(sink)
    window = np.hanning(len(source))
    basis = np.exp(-2j * np.pi * freq * np.arange(len(source)) / fs)
    source_value = np.dot(source * window, basis)
    sink_value = np.dot(sink * window, basis)

    if np.abs(source_value) < 1e-12:
        return np.nan, np.nan

    response = sink_value / source_value
    return np.abs(response), np.angle(response)


def load_channel_wav(wav_path, channel_names):
    """Load a PCM WAV as a mapping from configured names to channel waveforms."""
    with wave.open(wav_path, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        fs = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if n_channels != len(channel_names):
        raise ValueError(
            f"parameters.json lists {len(channel_names)} channel(s) but "
            f"{wav_path} has {n_channels} channel(s)"
        )

    scales = {1: (np.uint8, 128.0), 2: (np.int16, 32767.0), 4: (np.int32, 2147483647.0)}
    if sampwidth not in scales:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes in {wav_path}")
    dtype, scale = scales[sampwidth]
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if sampwidth == 1:
        data = (data - 128.0) / scale
    else:
        data /= scale
    data = data.reshape(-1, n_channels)
    return {name: data[:, index] for index, name in enumerate(channel_names)}, fs


def get_sample_files(samples_dir):
    """Return sorted (frequency, path) pairs for sample_*.wav files."""
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    samples = []
    for filename in os.listdir(samples_dir):
        if filename.startswith("sample_") and filename.endswith(".wav"):
            try:
                frequency = float(filename[len("sample_"):-len(".wav")])
            except ValueError:
                continue
            samples.append((frequency, os.path.join(samples_dir, filename)))

    if not samples:
        raise FileNotFoundError(f"No valid sample_*.wav files found in {samples_dir}")
    return sorted(samples)


def analyze_samples(input_dir, analysis_dir=None):
    """Analyze every configured WAV channel and source/sink resonance ratio."""
    analysis_dir = analysis_dir or os.path.join(input_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    settings = load_parameters(input_dir)
    channel_settings = sorted(settings["channels"], key=lambda channel: channel.get("id", 0))
    channel_names = [channel["name"] for channel in channel_settings]
    resonances = settings.get("compute_resonance", [])
    resonance_labels = [resonance_label(resonance) for resonance in resonances]
    sample_files = get_sample_files(os.path.join(input_dir, "samples"))

    n_samples = len(sample_files)
    freqs = np.array([frequency for frequency, _ in sample_files], dtype=float)
    channel_rms_dbfs = {name: np.zeros(n_samples) for name in channel_names}
    resonance_magnitudes = {label: np.zeros(n_samples) for label in resonance_labels}
    resonance_phases = {label: np.zeros(n_samples) for label in resonance_labels}

    report_path = os.path.join(analysis_dir, "report.csv")
    with open(report_path, "w", newline="") as report_file:
        writer = csv.writer(report_file)
        header = ["frequency_hz"] + [f"{name}_rms" for name in channel_names]
        for label in resonance_labels:
            header.extend((f"{label}_magnitude", f"{label}_phase_radians", f"{label}_phase_degrees"))
        writer.writerow(header)

        for index, (frequency, wav_path) in enumerate(sample_files):
            channels, fs = load_channel_wav(wav_path, channel_names)
            row = [frequency]
            rms_parts = []
            for name in channel_names:
                rms = float(np.sqrt(np.mean(channels[name] ** 2)))
                dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf
                channel_rms_dbfs[name][index] = dbfs
                row.append(rms)
                rms_parts.append(f"{name}_rms = {dbfs:5.2f} dBFS")

            resonance_parts = []
            for resonance, label in zip(resonances, resonance_labels):
                magnitude, phase = bin_ratio(
                    channels[resonance["source"]], channels[resonance["sink"]], frequency, fs
                )
                resonance_magnitudes[label][index] = magnitude
                resonance_phases[label][index] = phase
                row.extend((magnitude, phase, np.degrees(phase)))
                resonance_parts.append(
                    f"{label} |H| = {magnitude:5.2f} phase = {np.degrees(phase):7.2f} deg"
                )

            writer.writerow(row)
            print(f"{frequency:7.2f} Hz  " + "  ".join(resonance_parts + rms_parts))

    return freqs, channel_names, channel_rms_dbfs, resonance_labels, resonance_magnitudes, resonance_phases


def compute_average_magnitude_db(freqs, magnitudes):
    """Compute the mean transfer-function magnitude in dB across valid points."""
    valid = np.isfinite(freqs) & np.isfinite(magnitudes) & (magnitudes > 0)
    return float(np.mean(20 * np.log10(magnitudes[valid]))) if np.any(valid) else np.nan


def save_average_magnitude_csv(output_dir, experiment_name, resonance_averages):
    """Write average magnitude for each configured resonance."""
    output_path = os.path.join(output_dir, "average_magnitude.csv")
    with open(output_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("experiment_name", "resonance_name", "average_magnitude_db"))
        for resonance_name, average in resonance_averages.items():
            writer.writerow((experiment_name, resonance_name, average))
    return output_path


def compute_band_magnitudes(freqs, magnitudes, bands):
    """Compute the mean transfer-function magnitude in dB for each frequency band."""
    valid = np.isfinite(freqs) & np.isfinite(magnitudes) & (magnitudes > 0)
    results = {}
    for band_name, (low, high) in bands.items():
        in_band = valid & (freqs >= low) & (freqs < high)
        results[band_name] = (
            20 * np.log10(float(np.mean(magnitudes[in_band]))) if np.any(in_band) else float("nan")
        )
    return results


def save_band_magnitudes_csv(output_dir, band_results_by_resonance, bands):
    """Write per-band magnitude data for every configured resonance."""
    output_path = os.path.join(output_dir, "band_magnitudes.csv")
    with open(output_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("resonance_name", "band_name", "low_hz", "high_hz", "sum_magnitude_db"))
        for resonance_name, results in band_results_by_resonance.items():
            for band_name, magnitude in results.items():
                writer.writerow((resonance_name, band_name, *bands[band_name], magnitude))
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze recorded frequency sweep audio samples and write report CSVs."
    )
    parser.add_argument(
        "--input-dir", "--output-dir", default="default",
        help="Name or path of the experiment directory (e.g. experiments/<input-dir>).",
    )
    args = parser.parse_args()

    if os.path.isdir(args.input_dir):
        input_dir = args.input_dir
        experiment_name = os.path.basename(os.path.normpath(input_dir))
    else:
        input_dir = os.path.join("experiments", args.input_dir)
        experiment_name = args.input_dir

    samples_dir = os.path.join(input_dir, "samples")
    if not os.path.isdir(samples_dir) or not any(name.endswith(".wav") for name in os.listdir(samples_dir)):
        print(f"Error: No sample WAV files found in {samples_dir}", file=sys.stderr)
        sys.exit(1)

    analysis_dir = os.path.join(input_dir, "analysis")
    print(f"Analyzing recorded samples from {samples_dir}...")
    freqs, _, _, labels, magnitudes, _ = analyze_samples(input_dir, analysis_dir)

    averages = {label: compute_average_magnitude_db(freqs, magnitudes[label]) for label in labels}
    print("\nAverage magnitude:")
    for label, average in averages.items():
        print(f"  {experiment_name} [{label}]: {average:.6f} dB")
    print(f"Saved average magnitude to: {save_average_magnitude_csv(analysis_dir, experiment_name, averages)}")

    band_results = {label: compute_band_magnitudes(freqs, magnitudes[label], resonance_bands) for label in labels}
    print(f"\nBand magnitudes (mean |H| in band, dB):")
    for label, results in band_results.items():
        print(f"  [{label}]")
        for band_name, magnitude in results.items():
            low, high = resonance_bands[band_name]
            value = f"{magnitude:.4f} dB" if np.isfinite(magnitude) else "no data"
            print(f"    {band_name:30s} [{low:5g}-{high:5g} Hz]: {value}")
    print(f"Saved band magnitudes to: {save_band_magnitudes_csv(analysis_dir, band_results, resonance_bands)}")

    print(f"\nRun plot.py to generate plots from {analysis_dir}/report.csv")


if __name__ == "__main__":
    main()
