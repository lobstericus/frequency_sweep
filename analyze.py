"""
Stepped-sine resonance analysis.

Processes recorded audio files (.wav) from experiments/<input-dir>/samples/,
extracts the test frequency from the sample filename (sample_xxxxxx.x.wav),
computes transfer function magnitude and phase using a single-frequency DFT,
calculates RMS levels, and outputs to experiments/<input-dir>/analysis/:
  1. report.csv           - frequency, magnitude, phase, piezo_rms, mic_rms
  2. average_magnitude.csv - mean transfer-function magnitude (dB) summary
"""

import argparse
import csv
import os
import sys
import wave
import numpy as np


resonance_bands = {
    "A0": (270, 280),
    "A1": (460, 470),
    "B1-": (440, 470),
    "B1+": (470, 540),
    "Bridge hill": (2000, 3000),
    "Secondary/nasal hill": (4500, 6000),
}

# ---------------------------------------------------------------------


def bin_ratio(piezo, mic, freq, fs):
    """
    Removes DC bias, applies a Hann window, computes the single-frequency
    discrete Fourier transform (DFT) at the exact test frequency `freq`,
    and returns the complex transfer function ratio mic/piezo (magnitude, phase).
    """
    piezo = piezo - np.mean(piezo)
    mic = mic - np.mean(mic)

    window = np.hanning(len(piezo))
    piezo_w = piezo * window
    mic_w = mic * window

    t = np.arange(len(piezo_w)) / fs
    basis = np.exp(-2j * np.pi * freq * t)

    piezo_val = np.dot(piezo_w, basis)
    mic_val = np.dot(mic_w, basis)

    if np.abs(piezo_val) < 1e-12:
        return np.nan, np.nan

    H = mic_val / piezo_val
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


def analyze_samples(input_dir, analysis_dir=None):
    """
    Process recorded audio samples from input_dir/samples.
    Computes DFT magnitude/phase, RMS levels, writes report.csv in analysis_dir,
    and returns (freqs, magnitudes, phases, piezo_dbfs_arr, mic_dbfs_arr).
    """
    if analysis_dir is None:
        analysis_dir = os.path.join(input_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    samples_dir = os.path.join(input_dir, "samples")
    sample_files = get_sample_files(samples_dir)

    n_samples = len(sample_files)
    freqs = np.array([f for f, _ in sample_files], dtype=float)
    magnitudes = np.zeros(n_samples)
    phases = np.zeros(n_samples)
    piezo_dbfs_arr = np.zeros(n_samples)
    mic_dbfs_arr = np.zeros(n_samples)

    report_path = os.path.join(analysis_dir, "report.csv")
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


def compute_average_magnitude_db(freqs, magnitudes):
    """Compute the mean transfer-function magnitude in dB across all valid frequency points."""
    valid = np.isfinite(freqs) & np.isfinite(magnitudes) & (magnitudes > 0)
    if not np.any(valid):
        return np.nan
    return float(np.mean(20 * np.log10(magnitudes[valid])))


def save_average_magnitude_csv(output_dir, experiment_averages):
    """Write mean magnitude (dB) for each experiment to average_magnitude.csv."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "average_magnitude.csv")

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["experiment_name", "average_magnitude_db"])
        for experiment_name, avg in experiment_averages.items():
            writer.writerow([experiment_name, avg])

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
        in_band = valid & (freqs >= low) & (freqs <= high)
        if not np.any(in_band):
            results[band_name] = float("nan")
        else:
            total_linear = float(np.sum(magnitudes[in_band]))
            results[band_name] = 20 * np.log10(total_linear)
    return results


def save_band_magnitudes_csv(output_dir, band_results, bands):
    """
    Write per-band summed magnitude (dB) to band_magnitudes.csv.

    Parameters
    ----------
    output_dir   : destination directory
    band_results : dict mapping band_name -> sum_db (from compute_band_magnitudes)
    bands        : original resonance_bands dict (used to record Hz ranges)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "band_magnitudes.csv")

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["band_name", "low_hz", "high_hz", "sum_magnitude_db"])
        for band_name, sum_db in band_results.items():
            low, high = bands[band_name]
            writer.writerow([band_name, low, high, sum_db])

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
    freqs, mags, phases, piezo_dbfs_arr, mic_dbfs_arr = analyze_samples(input_dir, analysis_dir)

    experiment_averages = {experiment_name: compute_average_magnitude_db(freqs, mags)}
    average_magnitude_csv = save_average_magnitude_csv(analysis_dir, experiment_averages)

    print(f"\nAverage magnitude:")
    for exp_name, avg in experiment_averages.items():
        print(f"  {exp_name}: {avg:.6f} dB")
    print(f"Saved average magnitude to: {average_magnitude_csv}")

    band_results = compute_band_magnitudes(freqs, mags, resonance_bands)
    band_csv = save_band_magnitudes_csv(analysis_dir, band_results, resonance_bands)

    print(f"\nBand magnitudes (sum of |H| in band, dB):")
    for band_name, sum_db in band_results.items():
        low, high = resonance_bands[band_name]
        db_str = f"{sum_db:.4f} dB" if not np.isnan(sum_db) else "no data"
        print(f"  {band_name:30s} [{low:5g}–{high:5g} Hz]: {db_str}")
    print(f"Saved band magnitudes to: {band_csv}")

    print(f"\nRun plot.py to generate plots from {analysis_dir}/report.csv")