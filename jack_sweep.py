"""
Stepped-sine resonance measurement — PipeWire/JACK version.

    pw-jack python jack_sweep.py

Runs as a native JACK client (via pipewire-jack) instead of going through
PortAudio/ALSA. This means it shows up as its own node in qpwgraph/Helvum,
patches cleanly alongside Ardour, and avoids fighting PipeWire for
exclusive ALSA access to the interface.

Requires: pip install JACK-Client
(uses whatever libjack.so pipewire-jack provides — no separate JACK
server needs to be running under PipeWire)

Workflow per frequency step is unchanged from the PortAudio version:
  1. Drive the exciter at f for a settle period + capture period
  2. Record piezo (reference/input) and mic (response/output) simultaneously
  3. Remove DC bias
  4. Apply Hann window
  5. rfft both signals
  6. Take magnitude/phase at (or near) the bin closest to f
  7. H(f) = mic_fft(f) / piezo_fft(f)
  8. Store magnitude(f) and phase(f)
"""

import argparse
import csv
import os
import sys
import threading
import numpy as np
import jack
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Config — fill in PIEZO_IN_PORT / MIC_IN_PORT / EXCITER_OUT_PORT after
# running `python3 resonance_sweep_jack.py --list-ports` once to see
# the exact port names PipeWire exposes for the ZEDi10. They may not
# literally be "ZEDi10:..." depending on how the node got named.
# ---------------------------------------------------------------------
AUTO_CONNECT = True                      # False = patch manually in qpwgraph
EXCITER_OUT_PORT_L = "ZEDi10 Analog Surround 4.0:playback_FL"
EXCITER_OUT_PORT_R = "ZEDi10 Analog Surround 4.0:playback_FR"
PIEZO_IN_PORT = "ZEDi10 Analog Surround 4.0:capture_FL"
MIC_IN_PORT = "ZEDi10 Analog Surround 4.0:capture_FR"

SETTLE_TIME = 0.3   # seconds
CAPTURE_TIME = 0.5  # seconds
MIN_FREQ = 80
MAX_FREQ = 6000
NUM_SAMPLES = 100
RESPONSE_AREA_BASELINE_DB = -40


def list_ports():
    """Prints all JACK/PipeWire ports. Run this first to find the exact
    names for your ZEDi10 channels."""
    c = jack.Client("port_lister", no_start_server=True)
    print("Inputs (things you can send audio TO):")
    for p in c.get_ports(is_input=True):
        print(f"  {p.name}")
    print("\nOutputs (things you can capture FROM):")
    for p in c.get_ports(is_output=True):
        print(f"  {p.name}")
    c.close()


class SweepClient:
    """
    Wraps a single JACK client used across the whole sweep: one exciter
    output port, two capture input ports (piezo, mic).

    Between frequency steps, run_step() swaps in a new tone buffer and
    fresh capture buffers; the realtime process() callback just
    pulls/pushes samples until the step's capture buffers are full,
    then sets an Event the main thread waits on.
    """

    def __init__(self, auto_connect=True):
        self.client = jack.Client("resonance_sweep")
        self.out_port = self.client.outports.register("exciter_out")
        self.piezo_port = self.client.inports.register("piezo_in")
        self.mic_port = self.client.inports.register("mic_in")

        self.fs = self.client.samplerate
        self.blocksize = self.client.blocksize

        self._tone = np.zeros(0, dtype=np.float32)
        self._piezo_buf = np.zeros(0, dtype=np.float32)
        self._mic_buf = np.zeros(0, dtype=np.float32)
        self._play_idx = 0
        self._rec_idx = 0
        self._total_samples = 0
        self._done = threading.Event()

        self.client.set_process_callback(self._process)
        self.client.activate()

        if auto_connect:
            # one output port can fan out to both playback channels
            self.client.connect(self.out_port, EXCITER_OUT_PORT_L)
            self.client.connect(self.out_port, EXCITER_OUT_PORT_R)
            self.client.connect(PIEZO_IN_PORT, self.piezo_port)
            self.client.connect(MIC_IN_PORT, self.mic_port)
        else:
            print("AUTO_CONNECT is off — patch these in qpwgraph before "
                  "continuing:")
            print(f"  {self.client.name}:exciter_out -> exciter L")
            print(f"  {self.client.name}:exciter_out -> exciter R")
            print(f"  piezo source -> {self.client.name}:piezo_in")
            print(f"  mic source   -> {self.client.name}:mic_in")
            input("Press Enter once patched...")

    def _process(self, frames):
        out_buf = self.out_port.get_array()
        piezo_in = self.piezo_port.get_array()
        mic_in = self.mic_port.get_array()

        remaining_tone = len(self._tone) - self._play_idx
        n_play = max(0, min(frames, remaining_tone))
        if n_play > 0:
            out_buf[:n_play] = self._tone[self._play_idx:self._play_idx + n_play]
        if n_play < frames:
            out_buf[n_play:] = 0.0
        self._play_idx += n_play

        remaining_rec = self._total_samples - self._rec_idx
        n_rec = max(0, min(frames, remaining_rec))
        if n_rec > 0:
            end = self._rec_idx + n_rec
            self._piezo_buf[self._rec_idx:end] = piezo_in[:n_rec]
            self._mic_buf[self._rec_idx:end] = mic_in[:n_rec]
            self._rec_idx = end

        if self._total_samples > 0 and self._rec_idx >= self._total_samples:
            self._done.set()

    def run_step(self, freq, settle_time, capture_time):
        total_time = settle_time + capture_time
        n_total = int(total_time * self.fs)
        t = np.arange(n_total) / self.fs
        tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        self._tone = tone
        self._piezo_buf = np.zeros(n_total, dtype=np.float32)
        self._mic_buf = np.zeros(n_total, dtype=np.float32)
        self._total_samples = n_total
        self._play_idx = 0
        self._rec_idx = 0
        self._done.clear()

        self._done.wait()

        settle_samples = int(settle_time * self.fs)
        piezo = self._piezo_buf[settle_samples:]
        mic = self._mic_buf[settle_samples:]
        return piezo, mic

    def close(self):
        self.client.deactivate()
        self.client.close()


def bin_ratio(piezo, mic, freq, fs):
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


def save_channel_figure(freq, piezo, mic, fs, out_dir=None):
    """Saves a raw piezo/mic waveform plot for this step (to check
    levels/clipping). The time axis shows 10 full cycles at the current
    tone frequency, so low frequencies are displayed over a longer duration
    and high frequencies over a shorter duration."""
    if out_dir is None:
        out_dir = os.path.join("experiments", "default", "samples")

    cycles = 10
    total_time = cycles / freq if freq > 0 else 0
    n_points = max(1, min(len(piezo), int(total_time * fs)))
    start_idx = max(0, len(piezo) - n_points)
    piezo_slice = piezo[start_idx:]
    mic_slice = mic[start_idx:]
    t = np.arange(len(piezo_slice)) / fs

    fig, ax = plt.subplots()
    ax.plot(t, piezo_slice, label="Piezo (raw)")
    ax.plot(t, mic_slice, label="Mic (raw)")
    ax.axhline(1.0, color="r", ls="--", lw=0.8)
    ax.axhline(-1.0, color="r", ls="--", lw=0.8)
    ax.set_xlim(0, t[-1] if len(t) > 1 else 1.0)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Input channels @ {freq:.1f} Hz")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"sample_{freq:07.1f}.png"))
    plt.close(fig)


def load_experiment_csv(experiment_name):
    """Load a saved experiment report CSV and return arrays for plotting."""
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
                        test_name="default", output_dir=None, overlay=None):
    """Plot the swept transfer-function magnitude/phase and RMS traces."""
    if output_dir is None:
        output_dir = os.path.join("experiments", test_name)
    os.makedirs(output_dir, exist_ok=True)
    if overlay is None:
        overlay = []

    from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
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
    plt.show()


def sweep(freqs, settle_time=SETTLE_TIME, capture_time=CAPTURE_TIME,
          auto_connect=AUTO_CONNECT, test_name="default"):
    output_dir = os.path.join("experiments", test_name)
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Experiment directory already exists: {output_dir}. "
            f"Choose a new --test-name or remove the existing directory."
        )
    os.makedirs(output_dir, exist_ok=False)
    report_path = os.path.join(output_dir, "report.csv")

    sc = SweepClient(auto_connect=auto_connect)
    magnitudes = np.zeros(len(freqs))
    phases = np.zeros(len(freqs))
    piezo_dbfs_arr = np.zeros(len(freqs))
    mic_dbfs_arr = np.zeros(len(freqs))

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

        try:
            for i, f in enumerate(freqs):
                piezo, mic = sc.run_step(f, settle_time, capture_time)
                mag, ph = bin_ratio(piezo, mic, f, sc.fs)
                magnitudes[i] = mag
                phases[i] = ph
                piezo_rms = np.sqrt(np.mean(piezo ** 2))
                mic_rms = np.sqrt(np.mean(mic ** 2))
                piezo_dbfs = 20 * np.log10(piezo_rms) if piezo_rms > 0 else -np.inf
                mic_dbfs = 20 * np.log10(mic_rms) if mic_rms > 0 else -np.inf
                piezo_dbfs_arr[i] = piezo_dbfs
                mic_dbfs_arr[i] = mic_dbfs
                save_channel_figure(f, piezo, mic, sc.fs,
                                    out_dir=os.path.join(output_dir, "samples"))
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
        finally:
            sc.close()

    return magnitudes, phases, piezo_dbfs_arr, mic_dbfs_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep a resonant system and save a report.")
    parser.add_argument("--list-ports", action="store_true",
                        help="Print available JACK/PipeWire ports and exit.")
    parser.add_argument("--output-dir", default="default",
                        help="Name used for the output directory, e.g. experiments/<output-dir>.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run the live sweep and save a new experiment. If omitted, the script only plots existing data.")
    parser.add_argument("--plot", dest="plots", action="append", default=[],
                        metavar="EXPERIMENT_NAME",
                        help="Load and overlay a saved experiment from experiments/<EXPERIMENT_NAME>/report.csv. Can be passed multiple times.")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        sys.exit(0)

    output_dir = os.path.join("experiments", args.output_dir)

    if args.sweep:
        freqs = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), NUM_SAMPLES)
        mags, phases, piezo_dbfs_arr, mic_dbfs_arr = sweep(freqs, test_name=args.output_dir)
        overlay = []
        experiment_areas = {args.output_dir: compute_response_area(freqs, mags)}
    elif args.plots:
        source_experiment = args.plots[0]
        try:
            freqs, mags, phases, piezo_dbfs_arr, mic_dbfs_arr = load_experiment_csv(source_experiment)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        overlay = []
        experiment_areas = {source_experiment: compute_response_area(freqs, mags)}
        for experiment_name in args.plots[1:]:
            try:
                overlay_freqs, overlay_mags, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs = load_experiment_csv(experiment_name)
                overlay.append((experiment_name, overlay_freqs, overlay_mags, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs))
                experiment_areas[experiment_name] = compute_response_area(overlay_freqs, overlay_mags)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Error: {exc}")
                sys.exit(1)
    else:
        report_path = os.path.join(output_dir, "report.csv")
        if not os.path.exists(report_path):
            print(f"Error: experiment not found: {report_path}")
            print("Run with --sweep to generate it or choose an existing --output-dir.")
            sys.exit(1)

        freqs, mags, phases, piezo_dbfs_arr, mic_dbfs_arr = load_experiment_csv(args.output_dir)
        overlay = []
        experiment_areas = {args.output_dir: compute_response_area(freqs, mags)}

    if args.sweep and args.plots:
        for experiment_name in args.plots:
            try:
                overlay_freqs, overlay_mags, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs = load_experiment_csv(experiment_name)
                overlay.append((experiment_name, overlay_freqs, overlay_mags, overlay_phases, overlay_piezo_dbfs, overlay_mic_dbfs))
                experiment_areas[experiment_name] = compute_response_area(overlay_freqs, overlay_mags)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Error: {exc}")
                sys.exit(1)

    plot_response_curves(freqs, mags, phases, piezo_dbfs_arr, mic_dbfs_arr,
                        test_name=args.output_dir, output_dir=output_dir,
                        overlay=overlay)

    response_area_csv = save_response_area_csv(output_dir, experiment_areas)
    print(f"\nResponse area above {RESPONSE_AREA_BASELINE_DB} dB:")
    for experiment_name, area in experiment_areas.items():
        print(f"  {experiment_name}: {area:.6f} dB-Hz")
    print(f"Saved area summary to: {response_area_csv}")