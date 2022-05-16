from sklearn import cluster
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram

from yaml import SafeLoader
import yaml
from datetime import datetime
from typing import Any

import util
from spectrogram import quantize_spectrogram


# frequencies (in hertz) below this are filtered out
MIN_FREQ = 10**(-3/2)

# multiplicitive width of each frequency band
BANDWIDTH = 10**(1/2)

# number of frequency bands to use
# thus, max frequency (above which gets filtered out) is:
#   MIN_FREQ * BANDWIDTH**NUM_BANDS
NUM_BANDS = 6

# converts a vector's coordinate to the corresponding frequency band
DIM_TO_BAND = [
    (MIN_FREQ * BANDWIDTH**i, MIN_FREQ * BANDWIDTH**(i+1))
    for i in range(NUM_BANDS-1)
]


def axis_label(dimension: int) -> str:
    '''
    converts a (lower_freq, upper_freq) band to a string.
    for plot axes

    Parameters
    ----------
    dimension : int
        the vector coordinate that is being used

    Returns
    -------
    str
        axis label
    '''
    lower_freq = DIM_TO_BAND[dimension][0]
    upper_freq = DIM_TO_BAND[dimension][1]
    return f'{lower_freq:.3e} to {upper_freq:.3e} Hz RMS'


class ClusteredSpectrogram:
    '''
    vectorized spectrogram that can be clustered using scikit algorithms

    Attributes
    ----------
    spectrogram : Spectrogram
        a spectrogram
    vectors : np.ndarray
        a quantization of the spectrogram, turning each PSD into a
        NUM_BANDS-dimensional vector
    labels : list[int]
        the cluster that each vector in vectors belongs to

    Methods
    -------
    plot2D(dimension1, dimension2)
        creates a 2D scatter plot of the dimension1 coordinate vs the
        dimension2 coordinate of each vector in vectors, color-coded by labels
    '''
    def __init__(
        self, 
        timeseries_info_file: str, 
        cluster_method: type,
        cluster_method_params: dict[str, Any]
    ):
        '''
        Parameters
        ----------
        timeseries_info_file : str
            file specifying the details of the timeseries data to fetch (see
            timeseries-example.yml)
        cluster_method : type
            class from sklearn to use for clustering the vectors
        cluster_method_params : dict[str, Any]
            keyword arguments needed to instantiate the cluster_method class
        '''
        self.spectrogram = util.yaml_to_spectrogram(timeseries_info_file)

        self.vectors = quantize_spectrogram(
            self.spectrogram, 
            MIN_FREQ, 
            MIN_FREQ * BANDWIDTH**NUM_BANDS
        )

        self.labels = cluster_method(
            **cluster_method_params
        ).fit_predict(
            self.vectors
        )

    def plot2D(
        self, 
        dimension1: int, 
        dimension2: int,
        ax: plt.Axes
    ):
        '''
        creates a 2D scatter plot of the dimension1 coordinate vs the
        dimension2 coordinate of each vector in vectors, color-coded by
        the cluster labels

        Parameters
        ----------
        dimension1 : int
            the first coordinate of each vector in vectors that we want
        dimension2 : int
            the second coordinate of each vector in vectors that we want
        ax : plt.Axes
            plot coordinate system
        '''
        plot = ax.scatter(
            self.vectors[:,dimension1], 
            self.vectors[:,dimension2],
            c=self.labels,
        )

        ax.set_xlabel(axis_label(0))
        ax.set_ylabel(axis_label(1))
        ax.legend(*plot.legend_elements())

    def multiplot2D(
        self
    ):
        ''' plots adjacent dimensions against each other all in one plot '''
        _, axes = plt.subplots(1, int(NUM_BANDS/2))
        for i in range(int(NUM_BANDS/2)):
            self.plot2D(2*i, 2*i + 1, axes[i])