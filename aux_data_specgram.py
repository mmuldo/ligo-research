from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from datetime import datetime, timedelta

CHANNEL='L1:LSC-DARM_OUT_DQ' 
#START=1186736512
START=datetime.fromisoformat('2017-08-14 09:01:34')
END=datetime.fromisoformat('2017-08-14 12:00:46')
#END=1186747264
HOST='losc-nds.ligo.org'

def auxiliary_data_to_spectrogram(
    start: timedelta = timedelta(seconds=0), 
    end: timedelta = END - START,
    **kwargs
) -> Spectrogram:
    '''
    gets auxiliary data from a 3-hour window recorded by LIGO sensors.
    see https://www.gw-openscience.org/auxiliary/GW170814/

    Parameters
    ----------
    start_time : timedelta, optional
        mark to start at (default is a timedelta of 0 seconds)
    end_time : timedelta, optional
        mark to end at (default is a timedelta of around 3 hours)
    **kwargs
        keyword arguments to pass to TimeSeries.spectrogram function
        see https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.spectrogram

    Returns
    -------
    Spectrogram
        specgram of the aux data
    '''

    start_time = START + start
    end_time = START + end

    if start_time < START or end_time > END:
        raise ValueError('Start/end time out of range. ' +
                'Must be between 09:01:34 and 12:00:46 on 2017-08-14')

    return TimeSeries.fetch(
        channel=CHANNEL, 
        start=start_time, 
        end=end_time, 
        host=HOST
    ).spectrogram(**kwargs)


def main():
    specgram = auxiliary_data_to_spectrogram(stride=20, fftlength=8, overlap=4)

    #plot = specgram.plot(norm='log', vmin=1e-6, vmax=1e-4)
    #ax = plot.gca()
    #ax.set_ylim(10, 1000)
    #ax.set_yscale('log')
    #ax.colorbar(label='amplitude')
    #plot.show()
    return specgram

if __name__ == '__main__':
    specgram = main()
