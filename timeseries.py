from gwpy.timeseries import TimeSeries
from astropy.units import Unit
from astropy.units.quantity import Quantity
from dataclasses import dataclass


@dataclass
class TimeSeriesVector:
    vector: list[int]
    time: Quantity
    unit: Unit
