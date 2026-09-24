"""
Two damped harmonic oscillators coupled through a third, massed coupler
oscillator, with oscillator 1 sinusoidally driven.

Integrates the ODE for the three mass-spring-damper bodies (oscillator 1 and
oscillator 2 each connected to the coupler by a spring) using
scipy.integrate.solve_ivp, then plots each body's displacement and velocity
over time with matplotlib.
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Physical parameters

MASS_1      = 0.01    # kg
DAMPING_1   = 0.0001  # N*s/m  -- N / velocity
STIFFNESS_1 = 907.0   # N/m

MASS_2      = 0.03  # kg
DAMPING_2   = 1.7   # N*s/m   -- N / velocity
STIFFNESS_2 = 24000  # N/m


# Coupler (mass connecting oscillator 1 and oscillator 2)
MASS_3 = 0.005  # kg
DAMPING_3 = 0.0005  # N*s/m  -- N / velocity
#COUPLING_STIFFNESS = 150000  # N/m -- spring constant on each side of the coupler
COUPLING_STIFFNESS= 19000 # N/m -- spring constant on each side of the coupler .. less than calculated works 

# Driving force applied to oscillator 1

DRIVE_AMPLITUDE = 1.0  # N
DRIVE_FREQ = 400  # Hz
DRIVE_TYPE = "Sine"  # "Sine" or "Sawtooth"

# Initial conditions

X1_0 = 0.0  # m
V1_0 = 0.0  # m/s

X2_0 = 0.0  # m
V2_0 = 0.0  # m/s

X3_0 = 0.0  # m
V3_0 = 0.0  # m/s

# Simulation settings
T_MAX = 0.15  # s
NUM_POINTS = 1000


def sine_drive(t):
    """Returns the sinusoidal driving force applied to oscillator 1 at time(s) t."""
    return DRIVE_AMPLITUDE * np.sin(2 * np.pi * DRIVE_FREQ * t)


def sawtooth_drive(t):
    """Returns a zero-centered sawtooth driving force applied to oscillator 1."""
    phase = DRIVE_FREQ * np.asarray(t)
    return DRIVE_AMPLITUDE * (2 * (phase - np.floor(phase + 0.5)))


def natural_frequency_hz(mass, damping, stiffness):
    """Returns the damped natural (resonant) frequency, in Hz, of an isolated
    mass-spring-damper (ignoring coupling to the other oscillator). Returns 0
    if the system is overdamped or critically damped."""
    undamped_omega_sq = stiffness / mass
    decay_sq = (damping / (2 * mass)) ** 2
    if decay_sq >= undamped_omega_sq:
        return 0.0
    damped_omega = np.sqrt(undamped_omega_sq - decay_sq)
    return damped_omega / (2 * np.pi)


def rms(signal):
    """Returns the root-mean-square value of a signal array."""
    return np.sqrt(np.mean(np.square(signal)))


def oscillator_ode(t, state, coupling_stiffness, mass_3, damping_3, drive_function):
    """Returns [dx1/dt, dv1/dt, dx3/dt, dv3/dt, dx2/dt, dv2/dt] for oscillator 1
    and oscillator 2, each joined by a spring to a massed coupler (body 3) in
    between them, with oscillator 1 driven by a sinusoidal force."""

    x1, v1, x3, v3, x2, v2 = state

    drive = drive_function(t)

    a1 = (drive - DAMPING_1 * v1 - STIFFNESS_1 * x1 + coupling_stiffness * (x3 - x1)) / MASS_1

    a3 = (-damping_3 * v3 + coupling_stiffness * (x1 - x3) + coupling_stiffness * (x2 - x3)) / mass_3

    a2 = (-DAMPING_2 * v2 - STIFFNESS_2 * x2 + coupling_stiffness * (x3 - x2)) / MASS_2

    return [v1, a1, v3, a3, v2, a2]


def simulate(coupling_stiffness, mass_3, damping_3, drive_function=sine_drive):
    """Integrates the three bodies over [0, T_MAX] and returns (t, x1, v1, x3, v3, x2, v2) arrays."""
    t_eval = np.linspace(0, T_MAX, NUM_POINTS)
    solution = solve_ivp(
        oscillator_ode,
        t_span=(0, T_MAX),
        y0=[X1_0, V1_0, X3_0, V3_0, X2_0, V2_0],
        args=(coupling_stiffness, mass_3, damping_3, drive_function),
        t_eval=t_eval,
        method="RK45",
    )
    return (solution.t, solution.y[0], solution.y[1],
            solution.y[2], solution.y[3], solution.y[4], solution.y[5])


def draw_response(ax1, ax1_vel, ax3, ax3_vel, ax2, ax2_vel, coupling_stiffness,
                  mass_3, damping_3, drive_function=sine_drive, drive_name="Sine"):
    """Simulates with the given coupling stiffness, coupler mass, and coupler damping, and
    (re)draws each body's displacement/velocity (oscillator 1, coupler, oscillator 2) onto the given axes."""
    t, x1, v1, x3, v3, x2, v2 = simulate(coupling_stiffness, mass_3, damping_3, drive_function)

    for ax in (ax1, ax1_vel, ax3, ax3_vel, ax2, ax2_vel):
        ax.clear()

    ax1.plot(t, x1, color="tab:blue", label="displacement")
    ax1.set_ylabel("x (m)", color="tab:blue", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.yaxis.get_offset_text().set_fontsize(ax1.yaxis.get_offset_text().get_fontsize() / 2)
    x1_rms = rms(x1)
    ax1.set_title(f"Oscillator 1 (mass = {MASS_1:.3g} kg, x_rms = {x1_rms:.3g} m)", fontsize=10)
    ax1.grid(True)

    ax1_vel.plot(t, v1, color="tab:orange", label="velocity")
    ax1_vel.set_ylabel("v (m/s)", color="tab:orange", fontsize=10)
    ax1_vel.yaxis.tick_right()
    ax1_vel.yaxis.set_label_position("right")
    ax1_vel.tick_params(axis="y", labelcolor="tab:orange", labelsize=10)

    ax3.plot(t, x3, color="tab:blue", label="displacement")
    ax3.set_ylabel("x (m)", color="tab:blue", fontsize=10)
    ax3.tick_params(axis="y", labelcolor="tab:blue", labelsize=10)
    ax3.tick_params(axis="x", labelsize=10)
    ax3.yaxis.get_offset_text().set_fontsize(ax3.yaxis.get_offset_text().get_fontsize() / 2)
    x3_rms = rms(x3)
    ax3.set_title(f"Coupler (mass = {mass_3:.3g} kg, x_rms = {x3_rms:.3g} m)", fontsize=10)
    ax3.grid(True)

    ax3_vel.plot(t, v3, color="tab:orange", label="velocity")
    ax3_vel.set_ylabel("v (m/s)", color="tab:orange", fontsize=10)
    ax3_vel.yaxis.tick_right()
    ax3_vel.yaxis.set_label_position("right")
    ax3_vel.tick_params(axis="y", labelcolor="tab:orange", labelsize=10)

    ax2.plot(t, x2, color="tab:blue", label="displacement")
    ax2.set_xlabel("time (s)", fontsize=10)
    ax2.set_ylabel("x (m)", color="tab:blue", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=10)
    ax2.tick_params(axis="x", labelsize=10)
    ax2.yaxis.get_offset_text().set_fontsize(ax2.yaxis.get_offset_text().get_fontsize() / 2)
    x2_rms = rms(x2)
    ax2.set_title(f"Oscillator 2 (mass = {MASS_2:.3g} kg, x_rms = {x2_rms:.3g} m)", fontsize=10)
    ax2.grid(True)

    ax2_vel.plot(t, v2, color="tab:orange", label="velocity")
    ax2_vel.set_ylabel("v (m/s)", color="tab:orange", fontsize=10)
    ax2_vel.yaxis.tick_right()
    ax2_vel.yaxis.set_label_position("right")
    ax2_vel.tick_params(axis="y", labelcolor="tab:orange", labelsize=10)

    ax1.figure.suptitle(
        f"Coupled oscillators (coupling stiffness = {coupling_stiffness:.3g} N/m, "
        f"{drive_name.lower()} drive freq = {DRIVE_FREQ:.3g} Hz)",
        fontsize=12,
    )


def plot_response(output_path=None):
    """Builds the figure and, unless saving to a file, adds sliders to interactively
    adjust the coupling stiffness, coupler mass, and coupler damping, re-simulating on change."""
    if output_path is None and plt.get_backend().lower() == "agg":
        raise RuntimeError(
            "Interactive plotting requires a Matplotlib GUI backend. Install Tk support "
            "with 'sudo apt install python3-tk', then rerun this script. "
            "Use --output PATH to save a non-interactive PNG instead."
        )

    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, sharex=True, figsize=(19.2, 14.4), dpi=100)
    ax1_vel = ax1.twinx()
    ax3_vel = ax3.twinx()
    ax2_vel = ax2.twinx()

    drive_functions = {"Sine": sine_drive, "Sawtooth": sawtooth_drive}
    draw_response(
        ax1, ax1_vel, ax3, ax3_vel, ax2, ax2_vel, COUPLING_STIFFNESS, MASS_3, DAMPING_3,
        drive_functions[DRIVE_TYPE], DRIVE_TYPE,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.4)

    if output_path:
        fig.savefig(output_path)
        return

    fig.subplots_adjust(right=0.68, hspace=0.4)
    coupling_slider_ax = fig.add_axes((0.74, 0.15, 0.03, 0.7))
    coupling_slider = Slider(
        coupling_slider_ax, "Coupling\nStiffnes(N/m)", 0.0, 200000.0, valinit=COUPLING_STIFFNESS, orientation="vertical",
    )
    coupling_slider.label.set_fontsize(10)
    coupling_slider.valtext.set_fontsize(10)

    mass_slider_ax = fig.add_axes((0.82, 0.15, 0.03, 0.7))
    mass_slider = Slider(
        mass_slider_ax, "Coupler\nmass (kg)", 0.001, 0.10, valinit=MASS_3, orientation="vertical",
    )
    mass_slider.label.set_fontsize(10)
    mass_slider.valtext.set_fontsize(10)

    damping_slider_ax = fig.add_axes((0.90, 0.15, 0.03, 0.7))
    damping_slider = Slider(
        damping_slider_ax, "Coupler\ndamping\n(N*s/m)", 0.0, 0.002, valinit=DAMPING_3, orientation="vertical",
    )
    damping_slider.label.set_fontsize(10)
    damping_slider.valtext.set_fontsize(10)

    def on_change(_val):
        draw_response(
            ax1, ax1_vel, ax3, ax3_vel, ax2, ax2_vel,
            coupling_slider.val, mass_slider.val, damping_slider.val,
            drive_functions[DRIVE_TYPE], DRIVE_TYPE,
        )
        fig.canvas.draw_idle()

    coupling_slider.on_changed(on_change)
    mass_slider.on_changed(on_change)
    damping_slider.on_changed(on_change)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate and plot two oscillators coupled by a massed coupler.")
    parser.add_argument("--output", help="If given, save the plot to this path instead of displaying it")
    args = parser.parse_args()

    plot_response(args.output)


if __name__ == "__main__":
    main()
