"""
Audio capture via PipeWire/JACK for stepped-sine frequency measurements.

Handles opening the audio device, driving excitation tones, capturing
an arbitrary number of input channels (as defined in settings.json), and
saving raw waveform plots (.png) and PCM audio (.wav) to the experiment
samples directory.
"""

import argparse
import json
import os
import sys
import threading
import wave
import numpy as np
import jack
import matplotlib.pyplot as plt


DEFAULT_SETTINGS_PATH = "settings.json"


def load_settings(path):
    """Loads the sweep/capture configuration (port names, channel list,
    timing, and frequency range) from a JSON settings file."""
    with open(path) as f:
        return json.load(f)


def list_ports():
    """Prints all JACK/PipeWire ports. Run this first to find the exact
    names for your interface's channels."""
    c = jack.Client("port_lister", no_start_server=True)
    print("Inputs (things you can send audio TO):")
    for p in c.get_ports(is_input=True):
        print(f"  {p.name}")
    print("\nOutputs (things you can capture FROM):")
    for p in c.get_ports(is_output=True):
        print(f"  {p.name}")
    c.close()


def get_sample_rate():
    """Queries the JACK/PipeWire server's sample rate without registering any ports."""
    c = jack.Client("param_probe", no_start_server=True)
    fs = c.samplerate
    c.close()
    return fs


class SweepClient:
    """
    Wraps a single JACK client: one exciter output port, and N capture
    input ports/channels as described by `channels` (a list of
    {"name": ..., "port": ...} dicts).

    Between frequency steps, run_step() swaps in a new tone buffer and
    fresh capture buffers; the realtime process() callback just
    pulls/pushes samples until the step's capture buffers are full,
    then sets an Event the main thread waits on.
    """

    def __init__(self, channels, exciter_out_ports, auto_connect=True):
        self.channels = channels
        self.channel_names = [ch["name"] for ch in channels]
        self.exciter_out_ports = exciter_out_ports

        self.client = jack.Client("resonance_sweep")

        self.out_port = self.client.outports.register("exciter_out")
        self.in_ports = [
            self.client.inports.register(f"{ch['name']}_in") for ch in channels
        ]

        self.fs = self.client.samplerate
        self.blocksize = self.client.blocksize

        self._tone = np.zeros(0, dtype=np.float32)
        self._bufs = [np.zeros(0, dtype=np.float32) for _ in channels]

        self._play_idx = 0
        self._rec_idx = 0
        self._total_samples = 0
        self._done = threading.Event()

        self.client.set_process_callback(self._process)
        self.client.activate()

        if auto_connect:
            # one output port can fan out to multiple playback channels
            for out_port in self.exciter_out_ports:
                self.client.connect(self.out_port, out_port)
            for ch, in_port in zip(self.channels, self.in_ports):
                self.client.connect(ch["port"], in_port)
        else:
            print("AUTO_CONNECT is off — patch these in qpwgraph before "
                  "continuing:")
            for out_port in self.exciter_out_ports:
                print(f"  {self.client.name}:exciter_out -> {out_port}")
            for ch in self.channels:
                print(f"  {ch['name']} source -> {self.client.name}:{ch['name']}_in")
            input("Press Enter once patched...")

    def _process(self, frames):
        out_buf = self.out_port.get_array()
        ins = [p.get_array() for p in self.in_ports]

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
            for buf, in_arr in zip(self._bufs, ins):
                buf[self._rec_idx:end] = in_arr[:n_rec]
            self._rec_idx = end

        if self._total_samples > 0 and self._rec_idx >= self._total_samples:
            self._done.set()

    def run_step(self, freq, settle_time, capture_time):
        total_time = settle_time + capture_time
        n_total = int(total_time * self.fs)
        t = np.arange(n_total) / self.fs
        tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        self._tone = tone
        self._bufs = [np.zeros(n_total, dtype=np.float32) for _ in self.channels]
        self._total_samples = n_total
        self._play_idx = 0
        self._rec_idx = 0
        self._done.clear()

        self._done.wait()

        settle_samples = int(settle_time * self.fs)
        return [buf[settle_samples:] for buf in self._bufs]

    def close(self):
        self.client.deactivate()
        self.client.close()


def save_channel_wav(freq, channel_data, channel_names, fs, out_dir=None):
    """Saves an N-channel 16-bit PCM WAV file of the captured channels
    for this step, in the order given by channel_names."""
    if out_dir is None:
        out_dir = os.path.join("experiments", "default", "samples")

    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"sample_{freq:07.1f}.wav")
    stacked = np.column_stack(channel_data)
    pcm16 = (np.clip(stacked, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(len(channel_names))
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(pcm16.tobytes())


def save_channel_figure(freq, channel_data, channel_names, fs, out_dir=None):
    """Saves a raw waveform plot of all captured channels for this step (to
    check levels/clipping). The time axis shows 10 full cycles at the
    current tone frequency, so low frequencies are displayed over a longer
    duration and high frequencies over a shorter duration."""
    if out_dir is None:
        out_dir = os.path.join("experiments", "default", "samples")

    cycles = 10
    total_time = cycles / freq if freq > 0 else 0
    n_points = max(1, min(len(channel_data[0]), int(total_time * fs)))
    start_idx = max(0, len(channel_data[0]) - n_points)
    slices = [data[start_idx:] for data in channel_data]
    t = np.arange(len(slices[0])) / fs

    fig, ax = plt.subplots()
    for name, data_slice in zip(channel_names, slices):
        ax.plot(t, data_slice, label=f"{name} (raw)")
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


def capture_frequency_steps(freqs, channels, exciter_out_ports, output_dir="default",
                           settle_time=0.3, capture_time=0.5, auto_connect=True):
    """Captures audio steps across freqs and saves .wav and .png to the samples directory."""
    samples_dir = os.path.join("experiments", output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    channel_names = [ch["name"] for ch in channels]

    sc = SweepClient(channels, exciter_out_ports, auto_connect=auto_connect)
    try:
        for i, f in enumerate(freqs):
            channel_data = sc.run_step(f, settle_time, capture_time)
            save_channel_figure(f, channel_data, channel_names, sc.fs, out_dir=samples_dir)
            save_channel_wav(f, channel_data, channel_names, sc.fs, out_dir=samples_dir)
            print(f"[{i+1}/{len(freqs)}] {f:7.2f} Hz captured -> {samples_dir}")
    finally:
        sc.close()


def dump_parameters(output_dir, settings, num_samples):
    """Create a new file in the experiment output-dir directory called
    parameters.json and dumps the settings used for the sweep to it as JSON."""

    output_dir = os.path.join("experiments", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    params = dict(settings)
    params["num_samples"] = num_samples
    # id = 0-based index of the channel in the WAV file / channel_data lists.
    params["channels"] = [
        {**ch, "id": i} for i, ch in enumerate(settings["channels"])
    ]
    params["sample_rate"] = get_sample_rate()
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    # --settings is parsed first (and separately) since it determines the
    # defaults shown/used for the other arguments.
    settings_parser = argparse.ArgumentParser(add_help=False)
    settings_parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH,
                        help=f"Path to a settings JSON file (default: {DEFAULT_SETTINGS_PATH}).")
    settings_args, _ = settings_parser.parse_known_args()
    settings = load_settings(settings_args.settings)

    parser = argparse.ArgumentParser(
        parents=[settings_parser],
        description="Capture audio using PipeWire/JACK and save waveform plots and PCM WAV files."
    )
    parser.add_argument("--list-ports", action="store_true",
                        help="Print available JACK/PipeWire ports and exit.")
    parser.add_argument("--output-dir", default="default",
                        help="Name used for the output directory, e.g. experiments/<output-dir>.")
    parser.add_argument("--freq", type=float, default=None,
                        help="Single frequency to capture (Hz). If omitted, runs standard frequency sweep.")
    parser.add_argument("--num-samples", type=int, default=settings["num_samples"],
                        help=f"Number of frequency steps in the sweep (default: {settings['num_samples']}).")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        sys.exit(0)

    experiment_dir = os.path.join("experiments", args.output_dir)
    if os.path.exists(experiment_dir):
        print(f"Warning: experiment directory already exists: {experiment_dir}")
        sys.exit(1)

    min_freq = settings["min_freq"]
    max_freq = settings["max_freq"]

    if args.freq is not None:
        freqs = np.array([args.freq], dtype=float)
    else:
        freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), args.num_samples)

    freqs = np.round(freqs, 1)

    dump_parameters(args.output_dir, settings, num_samples=args.num_samples)

    capture_frequency_steps(
        freqs,
        channels=settings["channels"],
        exciter_out_ports=settings["exciter_out_ports"],
        output_dir=args.output_dir,
        settle_time=settings["settle_time"],
        capture_time=settings["capture_time"],
        auto_connect=settings["auto_connect"],
    )

