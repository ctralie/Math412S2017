# Math412S2017
Code for John's students for delay reconstruction and fast Rips code with "Ripser" (courtesy of Uli Bauer)

# Installation Instructions

To run this code, you will need to compile Ripser.  Go into the ``ripser'' directory, and type the following

~~~~~ bash
make ripser
make ripser-coeff
~~~~~

Now you should be able to run all of the examples:

* TDA.py: Code to compute persistent homology (basic command line wrapper around ripser, with functions to plot diagrams)

* SlidingWindow.py: Code to compute sliding window embeddings of 1D signals, with an example of cos(t) + cos(pi t), reporting persistent H1 and H2.  Also shows how to compute PCA

* HorseWhinnies.py: Quantifying periodicity and quasiperiodicity of snippets of audio from "horsewhinnie.wav"

* MusicSpeech.py: Running TDA on audio novelty functions to discover rhythmic periodicity in music.  Comparing journey.wav to speech.wav

* MusicFeatures.py: Quick and dirty music feature computation (MFCC, audio novelty functions) from spectrograms
