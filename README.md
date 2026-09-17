This code is used to compute resonance response of violins or other instruments.
It generates tones, sends them to a transducer and then records the audio from a mic.
We can compare the mic response to the transduce to get an idea of where the instrument resonances are.

The first step is to generate tones and capture the response to raw wav files

    jack_capture.py --output-dir <eperiment-name>

Then perform the analysis

    analyze.py --input-dir <experiment>

    