from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional
from yaml.loader import SafeLoader

import numpy as np
import yaml


def log_split(xs: np.ndarray, df: float, f0: float, base: float) -> np.ndarray:
    """Splits list into logarithmic bands

    Parameters
    ----------
    xs : np.ndarray
        quantized frequency spectrum, where each xs[i] corresponds to an
        amplitude

    df : float
        change in frequency between elements in xs

    f0 : float
        frequency associated with xs[0] (must be positive)

    base: float
        base of log

    Returns
    -------
    np.ndarray
        list of logarithmic bands
    """
    result = []

    # make sure first element isn't 0
    if f0 <= 0:
        raise ValueError( 
                f"First frequency of list must be positive, got {f0}")
    # list indexer
    i = 0 
    # current bandwidth; increase by base each time
    bandwidth = f0*base
    while i < len(xs):
        # current division
        band = []

        while i*df + f0 <= bandwidth and i < len(xs):
            band.append(xs[i])
            i += 1

        result.append(band)
        bandwidth *= base

    return np.array(result, dtype=object)
    

def quantize_psd(
    psd: np.ndarray, 
    df: float,
    f0: float,
    base: float
) -> np.ndarray:
    """Quantizes a PSD.

    The PSD is quantized into `log_base(fn/f0)` values, where fn is the highest
    frequency for the PSD.
    The value representing each band is the band's RMS^2. Since we are
    dealing with PSDs, this is given by simply computing the average value
    in the band.

    Parameters
    ----------
    psd : np.ndarray
        A Power Spectral Density, represented by an array of `floats`.

    df : float
        Change in frequency between each index of `psd`.

    f0 : float
        frequency associated with xs[0] (must be positive)

    base: float
        base of log

    Returns
    -------
    np.ndarray
        An array of log_base(fn/f0) values where each value is the RMS^2 of 
        the band it represents.
    """

    # split PSD into bands
    bands = log_split(psd, df, f0, base)

    return np.array([
        # RMS^2
        (1/len(band))*reduce(lambda x, y: x + y, band) 
        for band in bands])
    

def quantize_spectrogram(
    spectrogram: Spectrogram,
    low_frequency: Optional[float] = None,
    high_frequency: Optional[float] = None,
    base: float=10**(1/2)
) -> np.ndarray:
    """quantizes each PSD in a spectrogram

    Each PSD is quantized into `log_base(fn/f0)` values, where fn is the highest
    frequency for the PSD.
    The value representing each band is the band's RMS^2. Since we are
    dealing with PSDs, this is given by simply computing the average value
    in the band.

    Parameters
    ----------
    spectrogram : Spectrogram
        A spectrogram

    low_frequency : float, default=None
        lowest frequency to crop spectrogram to (default is None)

    high_frequency : float, default=None
        highest frequency to crop spectrogram to (default is None)

    base : float, default = 10**(1/2)
        base of log

    Returns
    -------
    np.ndarray
        An `m x n` matrix where `m = len(spectrogram)` (i.e. the number of
        timed samples) and `n = log_base(high_frequency/low_frequency)`
    """

    cropped = spectrogram.crop_frequencies( 
            low_frequency,
            high_frequency)

    df = cropped.df.value
    f0 = low_frequency if low_frequency else cropped.f0.value
    return np.array(
        [
            quantize_psd(
                np.array(psd, dtype=object), 
                df, 
                f0,
                base
            ) 
            for psd in cropped.data.tolist()
        ],
        dtype=float
    )
