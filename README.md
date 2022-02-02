# LIGO Noise Clustering

## Goals

* Correlate noise in sensor data with specific environmental sources.
* Identify noise states (ideally a small, discrete number) that LIGO takes on.
* Create training sets via clustering for supervised learning.

## Initial Steps

Using data from a single source (e.g. a ground seismometer), do the following:

1. Create a spectrogram of the data in one minute increments.  
    a. This spectrogram can then be thought of as a collection of frequency-
    dependent functions, indexed by time.
2. Quantize the spectrum at each minute into (e.g. 6) half decade bands.
    a. In other words, convert each frequency-depndent function into a vector
    (e.g. a 6D vector).
3. Cluster the resulting vectors.
