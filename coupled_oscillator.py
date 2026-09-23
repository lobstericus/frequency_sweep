"""
Two coupled damped harmonic oscillators, with oscillator 1 sinusoidally driven.

Integrates the ODE for two mass-spring-damper systems joined by a coupling
spring using scipy.integrate.solve_ivp, then plots both displacements and
velocities over time with matplotlib.
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Physical parameters

MASS_1      = 1.0  # kg
DAMPING_1   = 0.1  # N*s/m  -- N / velocity
STIFFNESS_1 = 2.0  # N/m

MASS_2      = 5.0  # kg
DAMPING_2   = 0.3  # N*s/m   -- N / velocity
STIFFNESS_2 = 20.0  # N/m

COUPLING_STIFFNESS = 6.22  # N/m

# Driving force applied to oscillator 1

DRIVE_AMPLITUDE = 1.0  # N
DRIVE_FREQ = 0.5  # Hz

# Initial conditions

X1_0 = 0.0  # m
V1_0 = 0.0  # m/s

X2_0 = 0.0  # m
V2_0 = 0.0  # m/s

# Simulation settings
T_MAX = 100.0  # s
NUM_POINTS = 3000


def drive_signal(t):
    """Returns the sinusoidal driving force applied to oscillator 1 at time(s) t."""
    return DRIVE_AMPLITUDE * np.sin(2 * np.pi * DRIVE_FREQ * t)


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


def oscillator_ode(t, state, coupling_stiffness):
    """Returns [dx1/dt, dv1/dt, dx2/dt, dv2/dt] for two mass-spring-dampers
    joined by a coupling spring, with oscillator 1 driven by a sinusoidal force."""

    x1, v1, x2, v2 = state

    drive = drive_signal(t)

    a1 = (drive - DAMPING_1 * v1 - STIFFNESS_1 * x1 + coupling_stiffness * (x2 - x1)) / MASS_1

    a2 = ( 0     -DAMPING_2 * v2 - STIFFNESS_2 * x2 + coupling_stiffness * (x1 - x2)) / MASS_2

    return [v1, a1, v2, a2]


def simulate(coupling_stiffness):
    """Integrates the coupled pair over [0, T_MAX] and returns (t, x1, v1, x2, v2) arrays."""
    t_eval = np.linspace(0, T_MAX, NUM_POINTS)
    solution = solve_ivp(
        oscillator_ode,
        t_span=(0, T_MAX),
        y0=[X1_0, V1_0, X2_0, V2_0],
        args=(coupling_stiffness,),
        t_eval=t_eval,
        method="RK45",
    )
    return solution.t, solution.y[0], solution.y[1], solution.y[2], solution.y[3]


def draw_response(ax_drive, ax1, ax1_vel, ax2, ax2_vel, coupling_stiffness):
    """Simulates with the given coupling stiffness and (re)draws the drive signal
    and each oscillator's displacement/velocity onto the given axes."""
    t, x1, v1, x2, v2 = simulate(coupling_stiffness)
    drive = drive_signal(t)

    for ax in (ax_drive, ax1, ax1_vel, ax2, ax2_vel):
        ax.clear()

    ax_drive.plot(t, drive, label="drive")
    ax_drive.set_ylabel("force (N)")
    ax_drive.set_title("Drive signal", fontsize="medium")
    ax_drive.legend()
    ax_drive.grid(True)

    ax1.plot(t, x1, color="tab:blue", label="displacement")
    ax1.set_ylabel("x (m)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    f1 = natural_frequency_hz(MASS_1, DAMPING_1, STIFFNESS_1)
    x1_rms = rms(x1)
    ax1.set_title(f"Oscillator 1 (f0 = {f1:.3g} Hz, x_rms = {x1_rms:.3g} m)", fontsize="medium")
    ax1.grid(True)

    ax1_vel.plot(t, v1, color="tab:orange", label="velocity")
    ax1_vel.set_ylabel("v (m/s)", color="tab:orange")
    ax1_vel.tick_params(axis="y", labelcolor="tab:orange")

    ax2.plot(t, x2, color="tab:blue", label="displacement")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("x (m)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    f2 = natural_frequency_hz(MASS_2, DAMPING_2, STIFFNESS_2)
    x2_rms = rms(x2)
    ax2.set_title(f"Oscillator 2 (f0 = {f2:.3g} Hz, x_rms = {x2_rms:.3g} m)", fontsize="medium")
    ax2.grid(True)

    ax2_vel.plot(t, v2, color="tab:orange", label="velocity")
    ax2_vel.set_ylabel("v (m/s)", color="tab:orange")
    ax2_vel.tick_params(axis="y", labelcolor="tab:orange")

    x_limit = max(np.abs(ax1.get_ylim()).max(), np.abs(ax2.get_ylim()).max())
    ax1.set_ylim(-x_limit, x_limit)
    ax2.set_ylim(-x_limit, x_limit)

    v_limit = max(np.abs(ax1_vel.get_ylim()).max(), np.abs(ax2_vel.get_ylim()).max())
    ax1_vel.set_ylim(-v_limit, v_limit)
    ax2_vel.set_ylim(-v_limit, v_limit)

    transfer_ratio = x2_rms / x1_rms if x1_rms else float("inf")
    ax_drive.figure.suptitle(
        f"Coupled oscillators (coupling stiffness = {coupling_stiffness:.3g} N/m, "
        f"x2_rms/x1_rms = {transfer_ratio:.3g})"
    )


def plot_response(output_path=None):
    """Builds the figure and, unless saving to a file, adds a slider to interactively
    adjust the coupling stiffness and re-simulate on change."""
    fig, (ax_drive, ax1, ax2) = plt.subplots(3, 1, sharex=True, dpi=300)
    ax1_vel = ax1.twinx()
    ax2_vel = ax2.twinx()

    draw_response(ax_drive, ax1, ax1_vel, ax2, ax2_vel, COUPLING_STIFFNESS)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        return

    fig.subplots_adjust(bottom=0.22)
    slider_ax = fig.add_axes((0.25, 0.05, 0.5, 0.03))
    slider = Slider(slider_ax, "Coupling (N/m)", 0.0, 20.0, valinit=COUPLING_STIFFNESS)

    def on_change(val):
        draw_response(ax_drive, ax1, ax1_vel, ax2, ax2_vel, val)
        fig.canvas.draw_idle()

    slider.on_changed(on_change)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate and plot two coupled damped oscillators.")
    parser.add_argument("--output", help="If given, save the plot to this path instead of displaying it")
    args = parser.parse_args()

    plot_response(args.output)


if __name__ == "__main__":
    main()
