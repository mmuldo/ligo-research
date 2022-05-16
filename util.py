from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from datetime import datetime, timedelta
from yaml.loader import SafeLoader
from typing import Optional

import yaml
import nds2

def yaml_to_spectrogram(
        filename: str,
        key: Optional[str] = None) -> Spectrogram:
    """reads in spectrogram parameters from yaml file

    Parameters
    ----------
    filename : str
        name of yaml file
    key : str, optional
        key of object to use within yaml file (default is None, in which
        case the yaml file itself is the parameters object)

    Returns
    -------
    Spectrogram
        a spectrogram of the given data
    """
    with open(filename) as f:
        if key:
            params = yaml.load(f, Loader=SafeLoader)[key]
        else:
            params = yaml.load(f, Loader=SafeLoader)

    return TimeSeries.fetch(
            start=datetime.fromisoformat(params['start']),
            end=datetime.fromisoformat(params['end']),
            **{
                k: v 
                for k, v in params.items() 
                if k not in [
                    'start', 
                    'end',
                    'stride',
                    'fftlength'
                ]
            }
            #channel=params['channel'],
            #host=params['host'],
            #port=params['port'],
            #connection=nds2.connection(params['host'],params['port'])
    ).spectrogram(
            stride=params['stride'],
            fftlength=params['fftlength'])
