"""
Convenience driver script: runs sweep_and_capture.py, analyze.py, and
plot.py in sequence for a single experiment.

Usage:
  python run_experiment.py my-experiment-name
  python run_experiment.py my-experiment-name --settings settings.json --num-samples 50
"""

import argparse
import subprocess
import sys


def run_step(description, cmd):
    print(f"\n=== {description}: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: '{description}' failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run sweep_and_capture.py, analyze.py, and plot.py for one experiment."
    )
    parser.add_argument("name", help="Experiment name, used as --output-dir/--input-dir for all three steps.")
    parser.add_argument("--settings", default=None,
                        help="Settings JSON file passed through to sweep_and_capture.py.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of frequency steps, passed through to sweep_and_capture.py.")
    parser.add_argument("--freq", type=float, default=None,
                        help="Single frequency to capture, passed through to sweep_and_capture.py.")
    parser.add_argument("--no-pw-jack", action="store_true",
                        help="Run sweep_and_capture.py directly instead of wrapping it with pw-jack.")
    args = parser.parse_args()

    capture_cmd = [] if args.no_pw_jack else ["pw-jack"]
    capture_cmd += [sys.executable, "sweep_and_capture.py", "--output-dir", args.name]
    if args.settings is not None:
        capture_cmd += ["--settings", args.settings]
    if args.num_samples is not None:
        capture_cmd += ["--num-samples", str(args.num_samples)]
    if args.freq is not None:
        capture_cmd += ["--freq", str(args.freq)]

    run_step("Capture", capture_cmd)
    run_step("Analyze", [sys.executable, "analyze.py", "--input-dir", args.name])
    run_step("Plot", [sys.executable, "plot.py", "--input-dir", args.name])

    print(f"\nDone: experiments/{args.name}")


if __name__ == "__main__":
    main()
