from sklearn import cluster, mixture
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
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

# frequencies (in hertz) above this are filtered out
MAX_FREQ = MIN_FREQ * BANDWIDTH ** NUM_BANDS

# converts a vector's coordinate to the corresponding frequency band
DIM_TO_BAND = [
    (MIN_FREQ * BANDWIDTH**i, MIN_FREQ * BANDWIDTH**(i+1))
    for i in range(NUM_BANDS)
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
    lower_freq = round(DIM_TO_BAND[dimension][0], 3)
    upper_freq = round(DIM_TO_BAND[dimension][1], 3)
    return f'{lower_freq} to {upper_freq} Hz RMS'

def get_centers(cluster_method: type, fitted_estimator: object) -> np.ndarray:
    '''returns a list of vectors that indicate the center of their respective
    cluster

    Parameters
    ----------
    cluster_method : type
        a class from sklearn
        currently supported are cluster.KMeans
    '''
    if cluster_method == cluster.KMeans:
        centers = fitted_estimator.cluster_centers_
    if cluster_method == mixture.GaussianMixture:
        centers = fitted_estimator.means_
    else:
        # if the passed cluster_method isn't supported, raise an error
        raise Exception(f'{cluster_method} is not supported by get_centers')

    return centers




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

    fitted_estimator : object
        the result of computing the clustering

    labels : list[int]
        the cluster that each vector in vectors belongs to

    centers : np.ndarray
        list of the NUM_BANDS-dimensional vectors that represent the center of 
        each of the clusters

    Methods
    -------
    plot2D(dimension1, dimension2)
        creates a 2D scatter plot of the dimension1 coordinate vs the
        dimension2 coordinate of each vector in vectors, color-coded by labels

    multiplot2D()
        plots adjacent dimensions against each other all in one plot
    '''
    def __init__(
        self, 
        spectrogram: Spectrogram, 
        cluster_method: type,
        cluster_method_params: dict[str, Any]
    ):
        '''
        Parameters
        ----------
        spectrogram : Spectrogram
            the spectrogram to cluster

        cluster_method : type
            class from sklearn to use for clustering the vectors

        cluster_method_params : dict[str, Any]
            keyword arguments needed to instantiate the cluster_method class
        '''
        self.spectrogram = spectrogram

        self.vectors = quantize_spectrogram(
            self.spectrogram, 
            MIN_FREQ, 
            MIN_FREQ * BANDWIDTH**NUM_BANDS,
            BANDWIDTH
        )

        self.fitted_estimator = cluster_method(
            **cluster_method_params
        ).fit(
            self.vectors
        )

        self.labels = self.fitted_estimator.predict(self.vectors)
        #self.labels = self.fitted_estimator.labels_
        
        self.centers = get_centers(cluster_method, self.fitted_estimator)
        #self.centers = self.fitted_estimator.cluster_centers_

    def plot2D(
        self, 
        dimension1: int, 
        dimension2: int,
        ax: plt.Axes
    ):
        '''
        creates a 2D scatter plot of the dimension1 coordinate vs the
        dimension2 coordinate of each vector in vectors, color-coded by
        the cluster labels; also plots cluster centers

        Parameters
        ----------
        dimension1 : int
            the first coordinate of each vector in vectors that we want

        dimension2 : int
            the second coordinate of each vector in vectors that we want

        ax : plt.Axes
            plot coordinate system
        '''
        # plot the vectors
        plot = ax.scatter(
            self.vectors[:,dimension1], 
            self.vectors[:,dimension2],
            c=self.labels,
            s=5,
        )

        # plot the vecter cluster centers
        ax.scatter(
            self.centers[:,dimension1], 
            self.centers[:,dimension2],
            c=[float(i) for i in range(len(self.centers))],
            marker='x',
            s=50,
        )

        ax.set_xlabel(axis_label(dimension1))
        ax.set_ylabel(axis_label(dimension2))
        ax.legend(*plot.legend_elements())

    def multiplot2D(
        self
    ):
        ''' plots adjacent dimensions against each other all in one plot '''
        _, axes = plt.subplots(1, int(NUM_BANDS/2))
        for i in range(int(NUM_BANDS/2)):
            self.plot2D(2*i, 2*i + 1, axes[i])


    def centers_plot(self):
        '''plots the quantized psd for each cluster center'''
        # x axis is the label of the cluster
        # each label should be centered in its vector
        x = [i + 0.5 for i in range(len(self.centers))]
        # the actual label names
        #xlabels = list(range(1, len(self.centers)+1))
        xlabels = set(self.labels)

        # y axis demarcates the frequency bands
        # given n frequency bands, we need n+1 frequency markings (an extra
        #   one for the highest frequency)
        y = list(range(len(self.centers[0])+1))
        # the actual labels are the frequencies themselves
        ylabels = [
            round(MIN_FREQ * BANDWIDTH**i, 3)
            for i in range(len(self.centers[0])+1)
        ]

        # generate figure and axes
        fig = plt.figure()
        ax = plt.axes()

        # generate pseudocolor plot
        # need to transpose centers, otherwise pcolormesh plots them
        #   transposed for some reason
        centers = self.centers.transpose()
        pcm = ax.pcolormesh(
            centers, 
            norm=LogNorm(vmin=centers.min(), vmax=centers.max()),
            edgecolors='k',
            linewidths=4,
        )

        # adjust axes tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels)

        # set axes labels with informative names
        ax.set_xlabel('state')
        ax.set_ylabel('frequency bands [Hz]')

        # generate colorbar
        fig.colorbar(
            pcm, 
            ax=ax,
            label='GW strain ASD [strain/$\sqrt{\mathrm{Hz}}$]',
        )

        # don't need grid
        plt.grid(visible=False)


    def spectrogram_plot(self):
        ''' plots spectrogram '''
        # get an initial plot of the spectrogram
        plot = self.spectrogram.plot(
            # use log scale for power
            norm='log', 
            # get extrema from the quantization (the vectors) of the spectrogram
            vmin=self.spectrogram.min().to_value(), 
            vmax=self.spectrogram.max().to_value()
        )

        # get plot axes
        ax = plot.gca()

        # narrow scope of frequencies
        ax.set_ylim(MIN_FREQ, MAX_FREQ)

        # use log scale for frequency axis
        ax.set_yscale('log')

        # colorbar label
        ax.colorbar(label='GW strain ASD [strain/$\sqrt{\mathrm{Hz}}$]')


    def vectors_plot(self):
        '''plots the vectors (quantized PSDs)'''
        # x axis is fine as is (minutes starting from 0)

        # y axis demarcates the frequency bands
        # given n frequency bands, we need n+1 frequency markings (an extra
        #   one for the highest frequency)
        y = list(range(len(self.centers[0])+1))
        # the actual labels are the frequencies themselves
        ylabels = [
            round(MIN_FREQ * BANDWIDTH**i, 3)
            for i in range(len(self.centers[0])+1)
        ]

        # generate figure and axes
        fig, ax = plt.subplots()

        # generate pseudocolor plot
        # need to transpose vectors, otherwise pcolormesh plots them
        #   transposed for some reason
        vectors = self.vectors.transpose()
        pcm = ax.pcolormesh(
            vectors, 
            norm=LogNorm(vmin=vectors.min(), vmax=vectors.max()),
        )

        # adjust axes tick labels
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels)

        # set axes labels with informative names
        ax.set_xlabel(f'Time [minutes]' 
                f' from {self.spectrogram.epoch.iso}')
        ax.set_ylabel('frequency bands [Hz]')

        # generate colorbar
        fig.colorbar(
            pcm, 
            ax=ax,
            label='GW strain ASD [strain/$\sqrt{\mathrm{Hz}}$]',
        )

    def states_plot(self):
        '''indicates the sequence of states (the cluster that the current 
        vector is apart of) over the sampled time'''
        # generate axes
        _, ax = plt.subplots()

        # plot the states (cluster labels) vs. time
        ax.plot(self.labels, drawstyle='steps-post')

        # adjust y axis ticks to only be on labels
        ax.set_yticks(list(set(self.labels)))

        # set axes labels with informative names
        ax.set_xlabel(f'Time [minutes]' 
                f' from {self.spectrogram.epoch.iso}')
        ax.set_ylabel('state')
