"""
Realtime oscilloscope for the JACK/PipeWire audio interface.

Opens the capture channels defined in settings.json (same port names used by
sweep_and_capture.py) into a circular buffer, and shows a live Qt waveform
plot with a positive-edge (rising) trigger, like a bench oscilloscope: the
display starts at the most recent negative-to-positive crossing of the
trigger level on the selected trigger channel.
"""

import argparse
import json
import sys
import threading
import time

import jack
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


DEFAULT_SETTINGS_PATH = "settings.json"
RING_SECONDS = 1.0
REFRESH_MS = 33  # ~30 fps
CURVE_COLORS = ("y", "c", "g", "m", "r", "w")


def load_settings(path):
    """Loads the channel configuration (port names) from a JSON settings file."""
    with open(path) as f:
        return json.load(f)


class ScopeClient:
    """JACK client: one input port per capture channel, plus one tone-generator output port.
    Each block of samples from the realtime process callback is written into a circular buffer,
    and the output port is driven with the generated tone (if enabled)."""

    TONE_AMPLITUDE = 0.5

    def __init__(self, channels, exciter_out_ports, auto_connect=True):
        self.channels = channels
        self.channel_names = [ch["name"] for ch in channels]
        self.exciter_out_ports = exciter_out_ports

        self.client = jack.Client("oscilloscope")
        self.in_ports = [
            self.client.inports.register(f"{ch['name']}_in") for ch in channels
        ]
        self.out_port = self.client.outports.register("tone_out")

        self.fs = self.client.samplerate
        self.ring_size = int(self.fs * RING_SECONDS)
        self.ring = np.zeros((len(channels), self.ring_size), dtype=np.float32)
        self.write_idx = 0

        self._tone_lock = threading.Lock()
        self._tone_on = False
        self._tone_freq = 440.0
        self._tone_waveform = "Sine"
        self._tone_phase = 0.0

        self.client.set_process_callback(self._process)
        self.client.activate()

        if auto_connect:
            for out_port in self.exciter_out_ports:
                try:
                    self.client.connect(self.out_port, out_port)
                except jack.JackError as exc:
                    print(f"  Warning: could not connect tone_out to {out_port}: {exc}", file=sys.stderr)
            for ch, in_port in zip(self.channels, self.in_ports):
                try:
                    self.client.connect(ch["port"], in_port)
                except jack.JackError as exc:
                    print(f"  Warning: could not connect {ch['port']}: {exc}", file=sys.stderr)
        else:
            print("AUTO_CONNECT is off — patch these in qpwgraph before continuing:")
            for out_port in self.exciter_out_ports:
                print(f"  {self.client.name}:tone_out -> {out_port}")
            for ch in self.channels:
                print(f"  {ch['name']} source -> {self.client.name}:{ch['name']}_in")
            input("Press Enter once patched...")

    def set_tone(self, on, freq, waveform):
        """Update the tone generator state from the GUI (thread-safe)."""
        with self._tone_lock:
            self._tone_on = on
            self._tone_freq = freq
            self._tone_waveform = waveform

    def _process(self, frames):
        out_buf = self.out_port.get_array()
        with self._tone_lock:
            on, freq, waveform = self._tone_on, self._tone_freq, self._tone_waveform
        if on:
            phase = self._tone_phase + freq * np.arange(frames) / self.fs
            if waveform == "Sawtooth":
                out_buf[:] = self.TONE_AMPLITUDE * (2 * (phase - np.floor(phase + 0.5)))
            else:
                out_buf[:] = self.TONE_AMPLITUDE * np.sin(2 * np.pi * phase)
            self._tone_phase = phase[-1] % 1.0
        else:
            out_buf[:] = 0.0

        end = self.write_idx + frames
        if end <= self.ring_size:
            for row, port in zip(self.ring, self.in_ports):
                row[self.write_idx:end] = port.get_array()
        else:
            first = self.ring_size - self.write_idx
            for row, port in zip(self.ring, self.in_ports):
                data = port.get_array()
                row[self.write_idx:] = data[:first]
                row[:end - self.ring_size] = data[first:]
        self.write_idx = end % self.ring_size

    def snapshot(self):
        """Returns an (n_channels, ring_size) array ordered oldest -> newest."""
        return np.concatenate(
            (self.ring[:, self.write_idx:], self.ring[:, :self.write_idx]), axis=1
        )

    def close(self):
        self.client.deactivate()
        self.client.close()


def find_trigger(signal, level=0.0):
    """Index of the most recent rising crossing of `level` in signal, or None."""
    crossings = np.flatnonzero((signal[:-1] < level) & (signal[1:] >= level)) + 1
    return int(crossings[-1]) if len(crossings) else None


class ScopeWindow(QtWidgets.QMainWindow):
    """Main window: pyqtgraph waveform pane plus trigger/timebase controls."""

    def __init__(self, client):
        super().__init__()
        self.client = client
        self.setWindowTitle("Oscilloscope")

        self.plot_widget = pg.PlotWidget(background="k")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.addLine(y=0, pen=pg.mkPen((128, 128, 128), style=QtCore.Qt.DashLine))
        self.plot_widget.addLegend()

        self.curves = [
            self.plot_widget.plot(pen=pg.mkPen(color, width=2), name=name)
            for color, name in zip(CURVE_COLORS, self.client.channel_names)
        ]

        self.trigger_channel = QtWidgets.QComboBox()
        self.trigger_channel.addItems(self.client.channel_names)

        self.level_spin = QtWidgets.QDoubleSpinBox(
            minimum=-1.0, maximum=1.0, singleStep=0.05, decimals=3
        )
        self.window_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.5, maximum=500.0, value=20.0, singleStep=1.0, suffix=" ms"
        )
        self.gain_spin = QtWidgets.QDoubleSpinBox(
            minimum=1.0, maximum=10.0, value=1.0, singleStep=0.5
        )

        controls = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(controls)

        self.channel_checkboxes = [QtWidgets.QCheckBox(name) for name in self.client.channel_names]
        for checkbox, curve in zip(self.channel_checkboxes, self.curves):
            checkbox.setChecked(True)
            checkbox.toggled.connect(curve.setVisible)
        self._rms_last_update = 0.0

        channels_group = QtWidgets.QGroupBox("Channels")
        channels_form = QtWidgets.QFormLayout(channels_group)
        for checkbox in self.channel_checkboxes:
            channels_form.addRow(checkbox)
        form.addRow(channels_group)

        scope_group = QtWidgets.QGroupBox("Oscilloscope")
        scope_form = QtWidgets.QFormLayout(scope_group)
        scope_form.addRow("Trigger channel", self.trigger_channel)
        scope_form.addRow("Trigger level", self.level_spin)
        scope_form.addRow("Time window", self.window_spin)
        scope_form.addRow("Gain", self.gain_spin)
        form.addRow(scope_group)

        self.tone_on = QtWidgets.QCheckBox("Tone on")
        self.tone_freq = QtWidgets.QDoubleSpinBox(
            minimum=50.0, maximum=5000.0, value=440.0, singleStep=10.0, suffix=" Hz"
        )
        self.tone_waveform = QtWidgets.QComboBox()
        self.tone_waveform.addItems(("Sine", "Sawtooth"))
        self.tone_output = QtWidgets.QComboBox()
        self.tone_output.addItems([port for port in self.client.exciter_out_ports])

        generator_group = QtWidgets.QGroupBox("Tone generator")
        generator_form = QtWidgets.QFormLayout(generator_group)
        generator_form.addRow(self.tone_on)
        generator_form.addRow("Frequency", self.tone_freq)
        generator_form.addRow("Waveform", self.tone_waveform)
        generator_form.addRow("Output", self.tone_output)
        form.addRow(generator_group)

        self._apply_tone()
        self.tone_on.toggled.connect(self._apply_tone)
        self.tone_freq.valueChanged.connect(self._apply_tone)
        self.tone_waveform.currentTextChanged.connect(self._apply_tone)
        self.tone_output.currentTextChanged.connect(self._reconnect_tone_output)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(controls)
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([220, 880])
        self.setCentralWidget(splitter)

        self.timer = QtCore.QTimer(self, timeout=self.update_display)
        self.timer.start(REFRESH_MS)

    def _apply_tone(self):
        self.client.set_tone(
            self.tone_on.isChecked(), self.tone_freq.value(), self.tone_waveform.currentText()
        )

    def _reconnect_tone_output(self):
        client = self.client.client
        for connection in client.get_all_connections(self.client.out_port):
            client.disconnect(self.client.out_port, connection.name)
        client.connect(self.client.out_port, self.tone_output.currentText())

    def update_display(self):
        data = self.client.snapshot()
        n = data.shape[1]
        window_samples = int(self.window_spin.value() * 1e-3 * self.client.fs)
        window_samples = min(window_samples, n // 2)

        trigger_signal = data[self.trigger_channel.currentIndex()]
        # Search the region that leaves room for the full display window.
        trigger_index = find_trigger(
            trigger_signal[: n - window_samples], level=self.level_spin.value()
        )
        if trigger_index is None:
            trigger_index = n - window_samples  # free run: show the latest data

        t = np.arange(window_samples) / self.client.fs
        stop = trigger_index + window_samples
        for curve, channel_data in zip(self.curves, data):
            curve.setData(t, channel_data[trigger_index:stop])
        self.plot_widget.setYRange(-1 / self.gain_spin.value(), 1 / self.gain_spin.value())

        now = time.monotonic()
        if now - self._rms_last_update >= 0.5:
            self._rms_last_update = now
            for name, checkbox, channel_data in zip(
                self.client.channel_names, self.channel_checkboxes, data
            ):
                rms = float(np.sqrt(np.mean(channel_data ** 2)))
                checkbox.setText(f"{name}  RMS={rms:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Realtime oscilloscope on the JACK capture channels."
    )
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH,
                        help="JSON settings file with the channel list (default: settings.json)")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    client = ScopeClient(
        settings["channels"], settings["exciter_out_ports"],
        auto_connect=settings.get("auto_connect", True),
    )

    app = QtWidgets.QApplication(sys.argv)
    window = ScopeWindow(client)
    window.resize(1100, 500)
    window.show()
    try:
        sys.exit(app.exec())
    finally:
        client.close()


if __name__ == "__main__":
    main()
