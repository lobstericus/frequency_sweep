"""
Stepped-sine resonance analysis.

Processes recorded audio files (.wav) from experiments/<input-dir>/samples/,
extracts the test frequency from the sample filename (sample_xxxxxx.x.wav),
computes transfer function magnitude and phase using a single-frequency DFT,
calculates RMS levels, and outputs to experiments/<input-dir>/analysis/:
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



# ---------------------------------------------------------------------
# Config & baseline
# ---------------------------------------------------------------------
RESPONSE_AREA_BASELINE_DB = -40


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

    experiment_areas = {experiment_name: compute_response_area(freqs, mags)}
    response_area_csv = save_response_area_csv(analysis_dir, experiment_areas)

    print(f"\nResponse area above {RESPONSE_AREA_BASELINE_DB} dB:")
    for exp_name, area in experiment_areas.items():
        print(f"  {exp_name}: {area:.6f} dB-Hz")
    print(f"Saved area summary to: {response_area_csv}")
    print(f"\nRun plot.py to generate plots from {analysis_dir}/report.csv")