from gwpy.timeseries import TimeSeries

# around 5 minutes of data
data = TimeSeries.fetch(
    'L1:LSC-DARM_OUT_DQ', 
    start=1186741600, 
    end=1186741950, 
    host='losc-nds.ligo.org'
)

specgram = data.spectrogram(20, fftlength=8, overlap=4)
plot = specgram.plot()
plot.show()
