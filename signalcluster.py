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
import yaml

LIGOTimeGPSParsable = Union[float, datetime, astropy.time.Time, str]

class BadHostFileError(BaseException):
    pass

try:
    with open('host.yml', 'r') as f:
        host_dict = yaml.load(f, Loader=yaml.Loader)
        HOST = host_dict['host']
        PORT = host_dict['port']
except FileNotFoundError:
    # default if no host file specified
    HOST = ''
    PORT = 0
except KeyError:
    raise BadHostFileError(f'check your host.yml file and make sure you specified the "host" and "port" keys')



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
    time: float,
    span: float,
    host: str = HOST,
    port: int = PORT,
) -> FrequencySeries:
    '''
    computes the psd for the timeseries taken from the given channel around the
    specified time frame

    Parameters
    ----------
    channel : str
        channel to grab the timeseries from
    time : float
        time (gps format) around which to compute the psd
    span : float
        number of seconds around time the the data will be taken at
    host : str, optional
        host of channel (default=HOST)
    port : str, optional
        port of channel at host (default=PORT)
    '''
    return TimeSeries.fetch(
        channel=channel,
        host=host,
        port=port,
        start=time-span,
        end=time+span
    ).psd()
