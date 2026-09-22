This code is used to compute resonance response of violins or other instruments.
It generates tones, sends them to a transducer and then records the audio from a mic.
We can compare the mic response to the transduce to get an idea of where the instrument resonances are.

We need to run python in the jack-pw pipewire shell so that it finds the jack provided by pipewire

The first step is to generate tones and capture the response to raw wav files

    jack-pw python jack_capture.py --output-dir <eperiment-name>

Then perform the analysis (does not use jack)

    python analyze.py --input-dir <experiment>

Finally you can create plots (does not use jack)

    python plot.py --input-dir <experiment> --output-dir <experiment>

You can combine multiple experiments in one plot

    jack-pw python plot.py \
        --input-dir <experiment1> \
        --input-dir <experiment2? \
        --output-dir <experiment>

