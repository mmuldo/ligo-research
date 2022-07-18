from __future__ import annotations
from gwpy.timeseries import TimeSeriesDict, TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.types.index import Index
from sklearn.cluster import OPTICS, DBSCAN, KMeans
from datetime import datetime
from typing import Union, Literal
import astropy.time
import astropy.units
import numpy as np
from dataclasses import dataclass
import yaml

LIGOTimeGPSParsable = Union[float, datetime, astropy.time.Time, str]

@dataclass
class TimeSeriesVectors:
    vectors: np.ndarray
    times: Index
    unit: astropy.units.Unit

    @classmethod
    def from_timeseriesdict(
        cls,
        tsdict: TimeSeriesDict
    ) -> TimeSeriesVectors:
        tslist = list(tsdict.values())

        assert len(tslist) > 0

        first_ts = tslist[0]

        return TimeSeriesVectors(
            vectors = np.array(tslist).T,
            # assuming everything worked correctly, the times and unit
            #   attributes should be the same for each time series
            times = first_ts.times,
            unit = first_ts.unit,
        )

    @classmethod
    def from_channel_list(
        cls,
        channels: list[str],
        host: str,
        port: int,
        start: LIGOTimeGPSParsable,
        end: LIGOTimeGPSParsable,
        verbose: bool
    ) -> TimeSeriesVectors:
        tsdict = TimeSeriesDict.fetch(
            channels=channels,
            host=host,
            port=port,
            start=start,
            end=end,
            verbose=verbose
        )

        return TimeSeriesVectors.from_timeseriesdict(tsdict)

    @classmethod
    def from_yaml(
        cls,
        filepath: str
    ) -> TimeSeriesVectors:
        with open(filepath, 'r') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)

        try:
            return TimeSeriesVectors.from_channel_list(**params)
        except ValueError as e:
            raise ValueError(f'Bad yaml file. {e.args[0]}')

    def cluster(
        self, 
        cluster_algorithm: SupportedClusterAlgorithms,
        *cluster_algorithm_args,
        **cluster_algorithm_kwargs
    ) -> SupportedClusterAlgorithms:

        clust = cluster_algorithm(
            *cluster_algorithm_args, 
            **cluster_algorithm_kwargs
        )

        clust.fit(self.vectors)

        return clust


def centers(
    clustered_data: SupportedClusterAlgorithms, 
    tsv: TimeSeriesVectors
) -> list[tuple[np.ndarray, Index]]:
    labeled_centers = {}

    for i in range(len(tsv.vectors)):
        label = clustered_data.labels_[i]
        vector = tsv.vectors[i]
        time = tsv.times[i]

        try:
            count, center, closest_vector_distance, closest_vector_time = labeled_centers[label]
            center = (center*count + vector) / (count + 1)
            count += 1
            closer = np.linalg.norm(vector - center) < closest_vector_distance
            closest_vector_distance = vector if closer else closest_vector_distance
            closest_vector_time = time if closer else closest_vector_time
        except KeyError:
            labeled_centers[label] = (
                0,
                vector,
                0,
                time
            )

    return [
        (center, closest_vector_time)
        for _, center, _, closest_vector_time in labeled_centers.values()
    ]

def timeseries_psd(
    ts: TimeSeries,
    time: Index
) -> FrequencySeries:
    ts.psd()

def yaml_to_timeseries(
    filepath: str
) -> TimeSeries:
    with open(filepath, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    return TimeSeries.fetch(**params)

def center_psd(ts, tsv):
    cs = centers(tsv.cluster(KMeans, 5), tsv)
    first = cs[0]
    plot = ts.psd().plot()
    plot.show()


def timeseriesdict_to_vectors(tsdict: TimeSeriesDict) -> np.ndarray:
    '''
    zips a collection of timeseries into vectors

    Parameters
    ----------
    tsdict : TimeSeriesDict
        collection of timeseries

    Returns
    -------
    np.ndarray
        an n-length collection of m-dimensional vectors where n is the number
        of samples in each timeseries from tsdict and m is the number of
        timeseries in tsdict
    '''
    return np.array(
        list(tsdict.values())
    ).T

def cluster(
    vectors: np.ndarray,
    cluster_algorithm: Literal[OPTICS, DBSCAN, KMeans],
    *cluster_algorithm_args,
    **cluster_algorithm_kwargs
) -> Union[OPTICS, DBSCAN, KMeans]:
    '''
    cluster a set of vectors using an algorithm from scikit-learn

    Parameters
    ----------
    vectors : np.ndarray
        vectors to cluster
    cluster_algorithm : type
        algorithm from scikit-learn to use; supported algorithms are OPTICS,
        DBSCAN, and KMeans
    *cluster_algorithm_args
        args to pass to cluster_algorithm
    **cluster_algorithm_kwargs
        kwargs to pass to cluster_algorithm

    Returns
    -------
    OPTICS, DBSCAN, or KMeans
        whatever clustering algorithm was used, with the data fitted
    '''
    clust = cluster_algorithm(
        *cluster_algorithm_args, 
        **cluster_algorithm_kwargs
    )

    clust.fit(vectors)

    return clust

def centers(
    vectors: np.ndarray,
    labels: list[int],
    n_clusters: int
) -> np.ndarray:
    '''
    calculate the centers of each cluster

    Parameters
    ----------
    vectors : np.ndarray
        the data that was clustered
    labels : list[int]
        the cluster labels
    n_clusters : int
        number of clusters

    Returns
    -------
    np.ndarray
        the cluster centers, of shape (n_clusters, vector dimension)
    '''
    # organize vectors by cluster label.
    # organized_data maps a cluster label (i.e. a list index) to a collection
    #   of vectors belonging to that cluster.
    organized_data = [np.empty(0) for _ in range(n_clusters)]
    for i in range(vectors.shape[0]):
        if labels[i] not in range(n_clusters):
            # skip data that isn't in a cluster
            continue

        # put the vector in its proper cluster
        np.append(organized_data[labels[i]], vectors[i])

    return np.array([ np.mean(clust, axis=0) for clust in organized_data ])

def times_of_centers(
    centers: np.ndarray,
    vectors: np.ndarray,
    times: Index
) -> list[quantity.Quantity]:
    '''
    get the times at which the vector closest to each cluster center occurred

    Parameters
    ----------
    centers : np.ndarray
        list of cluster centers
    vectors : np.ndarray
        the vector data
    times : Index
        list of times corresponding to each vector

    Returns
    -------
    list[quantity.Quatity]
        list of times corresponding to each cluster center
    '''
    # list in which each element indicates the index of the vector in vectors
    #   closest to the given center
    closest_indices = np.zeros(shape=centers.shape, dtype=int)

    for i in range(vectors.shape[0]):
        for j in range(centers.shape[0]):
            if np.linalg.norm(
                    centers[j] - vectors[i]
            ) < np.linalg.norm(
                centers[j] - vectors[closest_indices[j]]
            ):
                # if the current vector is closer to the current center than
                #   the current closest vector, update the closest vector
                closest_indices[j] = i

    # return the corresponding times of each closest vector
    return [times[index] for index in closest_indices]

def psd(
    channel: str,
    host: str,
    port: int,
    time: float,
    span: float
) -> FrequencySeries:
    '''
    computes the psd for the timeseries taken from the given channel around the
    specified time frame

    Parameters
    ----------
    channel : str
        channel to grab the timeseries from
    host : str
        host of channel
    port : str
        port of channel at host
    time : float
        time (gps format) around which to compute the psd
    span : float
        number of seconds around time the the data will be taken at
    '''
    return TimeSeries.fetch(
        channel=channel,
        host=host,
        port=port,
        start=time-span,
        end=time+span
    ).psd()
