from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional

import numpy as np

def log_split(xs: np.ndarray, df: float, f0: float, base: float) -> np.ndarray:
    """Splits list into half-decade bands """
    result = []

    # make sure first element isn't 0
    if f0 <= 0:
        raise ValueError(
                f"First frequency of list must be positive, got {f0}"
                )
    # list indexer
    i = 0 
    # current bandwidth; increase by sqrt(10) each time
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
        f0: float
        ) -> np.ndarray:
    """Quantizes a PSD into `n` values.

    The value representing each band is the band's RMS^2. Since we are
    dealing with PSDs, this is given by simply computing the average value
    in the band.

    Parameters
    ----------
    psd : `ndarray`
        A Power Spectral Density, represented by an array of `floats`.
    df : `float`
        Change in frequency between each index of `psd`.

    Returns
    -------
    `ndarray`
        An array of `n` values where each value is the RMS^2 of the band it
        represents.
    """

    bands = log_split(psd, df, f0, 10**(1/2))

    return np.array([
        # RMS^2
        (1/len(band))*reduce(lambda x, y: x + y, band) 
        for band in bands
        ])
    
def quantize_spectrogram(
        spectrogram: Spectrogram,
        low_frequency: Optional[float] = None,
        high_frequency: Optional[float] = None,
        ) -> np.ndarray:
    cropped = spectrogram.crop_frequencies(
            low_frequency,
            high_frequency
            )

    df = cropped.df.value
    f0 = low_frequency if low_frequency else cropped.f0.value
    return np.array(
            [quantize_psd(psd, df, f0) for psd in cropped.data.tolist()],
            dtype=object)


def main():
    CHANNEL='L1:LSC-DARM_OUT_DQ' 
    START=datetime.fromisoformat('2017-08-14 09:01:34')
    END=datetime.fromisoformat('2017-08-14 12:00:46')
    HOST='losc-nds.ligo.org'

    return {
            "specgram": TimeSeries.fetch(
                channel=CHANNEL, 
                start=START,
                end=START+timedelta(minutes=30),
                host=HOST
                ).spectrogram(stride=60, fftlength=60)
            }

if __name__ == "__main__":
    specgram = main()["specgram"]
    vectors = quantize_spectrogram(specgram, 0.01, 30)
