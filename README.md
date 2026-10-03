# Introduction

A collection of tools for analyzing instrument resonances.
The tools include an oscilloscope, a frequency sweep catpure utility,
a resonance analysis module and a plotting utility.

# Software setup
 * Currently implemented as python scripts using virtual environment
 * source venv/bin/activate

# Physical Setup

 * Setup the violin on a chin rest and a pillow under the scroll so that the whole body can vibrate freely.
 * Stick a cloth or tissue under strings to prevent sympathetic resonances
 * Clip the transducer to the bridge of the violin
 * Place the mic 30cm above the bridge.
 * Use the oscilloscope to setup the channels. Set the frequency to 440 and wave shape to sine and turn on the generator. Make sure the mic, exciter piezo and body piezo are all showing 0.6 RMS signal amplitude. You can adjust the direct injection box or audio interface gains.

# Sample capture
 * Use the sweep_and_capture.py module to grab a sound sample
 * It needs to be run with the pipewire jack wrapper

    pw-jack python sweep_and_capture.py --ouput-dir <instrument>-<date>-<trial>

# Resonance Analysis

The analyze.py module is used to compute resonance response of violins or other instruments.
We can compare the mic response to the transducer to get an idea of where the instrument resonances are.

    python analyze.py --input-dir <experiment>

# Plotting Results

Finally you can create plots (does not need pipewire jack)

    python plot.py --input-dir <experiment> --output-dir <experiment>

You can combine multiple experiments in one plot

    jack-pw python plot.py \
        --input-dir <experiment1> \
        --input-dir <experiment2> \
        --output-dir <experiment>

