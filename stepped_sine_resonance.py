"""
Stepped-sine resonance measurement.

Workflow per frequency step:
  1. Drive the exciter at f for a settle period + capture period
  2. Record piezo (reference/input) and mic (response/output) simultaneously
  3. Remove DC bias
  4. Apply Hann window
  5. rfft both signals
  6. Take magnitude/phase at (or near) the bin closest to f
  7. H(f) = mic_fft(f) / piezo_fft(f)
  8. Store magnitude(f) and phase(f)

Swap out `play_and_record()` for your actual audio I/O (sounddevice is
assumed below since it's the simplest cross-platform option for
simultaneous playback + multi-channel capture).
"""

import os

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # you can stub this out if using a different I/O backend


def list_devices():
    """
    Prints all available audio devices with their index, name, and
    max input/output channel counts. Run this first to find the
    index of your interface, and to confirm how many input/output
    channels it exposes.
    """
    print(sd.query_devices())


def make_tone(freq, duration, fs, amplitude=0.5):
    t = np.arange(int(duration * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


def play_and_record(freq, fs, settle_time, capture_time,
                     device, in_channels, out_channel):
    """
    Plays a sine tone on `out_channel` and records from `in_channels`
    simultaneously, on the given `device`. Returns (piezo, mic)
    arrays covering only the capture window (after settle_time has
    elapsed).

    device:       int index (same device for in+out), or a tuple
                  (input_device, output_device) if using two separate
                  interfaces. Use list_devices() to find indices.
    in_channels:  tuple of 2 channel numbers (1-indexed) for
                  (piezo, mic), e.g. (1, 2)
    out_channel:  channel number (1-indexed) the exciter is
                  connected to on this device, e.g. 1
    """
    total_time = settle_time + capture_time
    tone = make_tone(freq, total_time, fs)

    # sounddevice needs the output array shaped (samples, out_channels);
    # we only drive one channel, so build a single-column array and
    # tell it which physical output channel to map that column to.
    tone = tone.reshape(-1, 1)

    rec = sd.playrec(tone, samplerate=fs,
                      input_mapping=list(in_channels),
                      output_mapping=[out_channel],
                      device=device)
    sd.wait()

    settle_samples = int(settle_time * fs)
    piezo = rec[settle_samples:, 0]
    mic = rec[settle_samples:, 1]
    return piezo, mic


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

    # guard against a silent/zero reference bin
    if np.abs(piezo_fft[bin_idx]) < 1e-12:
        return np.nan, np.nan

    H = mic_fft[bin_idx] / piezo_fft[bin_idx]
    return np.abs(H), np.angle(H)


def sweep(freqs, fs, device, in_channels, out_channel,
          settle_time=0.3, capture_time=0.5):
    """
    Runs a full stepped-sine sweep across `freqs` and returns
    (magnitudes, phases) arrays aligned with `freqs`.
    """
    magnitudes = np.zeros(len(freqs))
    phases = np.zeros(len(freqs))

    for i, f in enumerate(freqs):
        piezo, mic = play_and_record(f, fs, settle_time, capture_time,
                                      device=device, in_channels=in_channels,
                                      out_channel=out_channel)
        mag, ph = bin_ratio(piezo, mic, f, fs)
        magnitudes[i] = mag
        phases[i] = ph
        print(f"{f:8.1f} Hz  |H| = {mag:.4f}  phase = {np.degrees(ph):6.1f} deg")

    return magnitudes, phases


if __name__ == "__main__":
    # ---- run list_devices() once to find your interface's index,
    # ---- then set these to match your setup ----
    list_devices()

    DEVICE = 3          # index from list_devices() output
    SAMPLE_RATE = 48000  # match what your interface/converters support
    IN_CHANNELS = (1, 2)  # (piezo input channel, mic input channel), 1-indexed
    OUT_CHANNEL = 1        # channel the exciter is wired to, 1-indexed

    # example: log-spaced sweep from 80 Hz to 4 kHz, 60 points
    freqs = np.logspace(np.log10(80), np.log10(4000), 60)

    mags, phases = sweep(freqs, fs=SAMPLE_RATE, device=DEVICE,
                          in_channels=IN_CHANNELS, out_channel=OUT_CHANNEL)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.semilogx(freqs, 20 * np.log10(mags))
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both")

    ax2.semilogx(freqs, np.degrees(phases))
    ax2.set_ylabel("Phase (deg)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", "resonance_response.png"))
    plt.show()
