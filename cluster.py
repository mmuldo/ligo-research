from sklearn.cluster import KMeans
from yaml import SafeLoader
from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from datetime import datetime

import yaml
import util
import spectrogram



specgram = util.yaml_to_spectrogram('time-series.yml')

vectors = spectrogram.vectorize(specgram, 10**(-3/2), 10**(3/2))

km = KMeans(n_clusters=4).fit(vectors)
