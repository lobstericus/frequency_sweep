"""
Audio capture via PipeWire/JACK for stepped-sine frequency measurements.

Handles opening the audio device, driving excitation tones, capturing
piezo/mic responses, and saving raw waveform plots (.png) and PCM audio (.wav)
to the experiment samples directory.
"""

import argparse
import os
import sys
import threading
import wave
import numpy as np
import jack
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Config — port names and timing parameters
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
    Wraps a single JACK client: one exciter output port, two capture input
    ports (piezo, mic).

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


def save_channel_wav(freq, piezo, mic, fs, out_dir=None):
    """Saves a 2-channel 16-bit PCM WAV file of the captured piezo (ch1)
    and mic (ch2) data for this step."""
    if out_dir is None:
        out_dir = os.path.join("experiments", "default", "samples")

    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"sample_{freq:07.1f}.wav")
    stereo = np.column_stack((piezo, mic))
    pcm16 = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(pcm16.tobytes())


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


def capture_frequency_steps(freqs, output_dir="default",
                           settle_time=SETTLE_TIME, capture_time=CAPTURE_TIME,
                           auto_connect=AUTO_CONNECT):
    """Captures audio steps across freqs and saves .wav and .png to the samples directory."""
    samples_dir = os.path.join("experiments", output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    sc = SweepClient(auto_connect=auto_connect)
    try:
        for i, f in enumerate(freqs):
            piezo, mic = sc.run_step(f, settle_time, capture_time)
            save_channel_figure(f, piezo, mic, sc.fs, out_dir=samples_dir)
            save_channel_wav(f, piezo, mic, sc.fs, out_dir=samples_dir)
            print(f"[{i+1}/{len(freqs)}] {f:7.2f} Hz captured -> {samples_dir}")
    finally:
        sc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture audio using PipeWire/JACK and save waveform plots and PCM WAV files."
    )
    parser.add_argument("--list-ports", action="store_true",
                        help="Print available JACK/PipeWire ports and exit.")
    parser.add_argument("--output-dir", "--ouptut-dir", default="default",
                        help="Name used for the output directory, e.g. experiments/<output-dir>.")
    parser.add_argument("--freq", type=float, default=None,
                        help="Single frequency to capture (Hz). If omitted, runs standard frequency sweep.")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        sys.exit(0)

    if args.freq is not None:
        freqs = np.array([args.freq], dtype=float)
    else:
        freqs = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), NUM_SAMPLES)

    capture_frequency_steps(freqs, output_dir=args.output_dir)
