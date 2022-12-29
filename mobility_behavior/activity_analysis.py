# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 17:45
# @Author  : Haoliang Chang
import os
from typing import Tuple
from collections import Counter
import numpy as np
import pandas as pd

# For Point-of-Interest (POI) analysis
from scipy.stats import entropy

# For spatial analysis
import geopandas as gpd

# For clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# For data analysis and configuration
from data_paths import records_path
from visualizations import plot_inertias
from utils import find_nearby_pois
from utils import hong_kong_epsg


def elbow_method(data_arr: np.array, highest_cluster_num: int = 15,
                 standardize_data: bool = True) -> dict:
    """
    Use elbow method to select the best cluster number. For reference, please check:
    https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    :param data_arr: a numpy array saving the data we want to cluster
    :param highest_cluster_num: the highest number of clusters considered in this study
    :param standardize_data: whether standardize the data before clustering or not
    :return: one dictionary saving the inertia values
    """
    inertias = []
    mapping = {}

    K = range(3, highest_cluster_num + 1)  # Specify the number of clusters we want to try (3 to 20)

    if standardize_data:
        standardized_scaler = StandardScaler()
        data_arr_transformed = standardized_scaler.fit_transform(data_arr)
    else:
        data_arr_transformed = data_arr

    for k in K:
        # Building and fitting the model
        print('Checking the cluster num: {}'.format(k))
        kmeanModel = KMeans(n_clusters=k, max_iter=10)
        kmeanModel.fit(data_arr_transformed)

        # Store the inertias
        inertia = kmeanModel.inertia_  # Sum of squared distances of samples to their closest cluster center
        inertias.append(inertia)

        # Create the result dictionary
        mapping[k] = inertia

    return mapping


def cluster_users(cluster_num: int, data_array: np.ndarray) -> np.ndarray:
    """
    Cluster the Twitter users based on number of tweets and time span
    :return: a numpy array saving the cluster labels
    """
    scaler = StandardScaler()
    transformed_data = scaler.fit_transform(data_array)
    k_means_model = KMeans(n_clusters=cluster_num, max_iter=10)
    k_means_model.fit(transformed_data)  # Cluster the hotspots given bandwidth_t = 3
    cluster_labels = k_means_model.labels_
    return cluster_labels


def get_poi_classes(poi_data: gpd.geodataframe) -> set:
    """
    Get the POI classes given a point-of-interest data
    :param poi_data: a Point-of-Interest dataframe
    :return: a python set saving the all the POI classes
    """
    assert 'fclass' in poi_data, 'The POI data should have a column fclass'
    return set(poi_data['fclass'])


def compute_poi_entropy(tweet_geo_lat: float, tweet_geo_lon: float,
                        poi_data: gpd.geodataframe.GeoDataFrame) -> float:
    """
    Compute the POI entropy for a geocoded tweet.
    :param tweet_geo_lat: the latitude of tweets
    :param tweet_geo_lon: the longitude of tweets
    :param poi_data: the Point-of-Interest (POI) information
    :return: the poi entropy for a geocoded tweet
    """
    assert 'fclass' in poi_data, "The Point-of-Interest information should have a column named 'fclass'"
    nearby_points = find_nearby_pois(geo_tweet_lat=tweet_geo_lat,
                                     geo_tweet_lon=tweet_geo_lon,
                                     poi_data=poi_data, point_epsg=hong_kong_epsg,
                                     set_point_4326=True)
    poi_class_cluster = Counter(nearby_points['fclass'])
    proportion_list = [poi_class_cluster[class_name] / sum(
        poi_class_cluster.values()) for class_name in poi_class_cluster]
    return entropy(proportion_list)


def poi_entropy_for_each_user(user_ids: set, tweet_dataframe: pd.DataFrame,
                              poi_dataframe: gpd.GeoDataFrame):
    """
    Compute the poi entropy for each user based on POI data and tweet dataframe
    :param user_ids: a set containing the ids of all the users
    :param tweet_dataframe: a pandas tweet dataframe
    :param poi_dataframe: a geopandas POI dataframe
    :return: a Python dictionary saving the POI entropy values for each user
    """
    assert 'user_id_str' in tweet_dataframe, \
        "The tweet data should have a column named 'user_id_str'"
    poi_entropy_results_dict = {}
    lat_colname, lon_colname = 'place_lat', 'place_lon'
    for user_id in user_ids:
        entropy_value_list = []
        select_tweets = tweet_dataframe.loc[
            tweet_dataframe['user_id_str'] == user_id].reset_index(drop=True)
        for index, row in select_tweets.iterrows():
            tweet_lat, tweet_lon = row[lat_colname], row[lon_colname]
            poi_entropy = compute_poi_entropy(tweet_geo_lat=tweet_lat,tweet_geo_lon=tweet_lon,
                                              poi_data=poi_dataframe)
            entropy_value_list.append(poi_entropy)
        poi_entropy_results_dict[user_id] = entropy_value_list
    return poi_entropy_results_dict


def infer_activity(tweet_geo: gpd.GeoDataFrame.geometry,
                   poi_data: gpd.geodataframe.GeoDataFrame):
    """

    :param tweet_geo:
    :param poi_data:
    :return:
    """
    pass


def main_user_cluster(mobility_before: pd.DataFrame, mobility_during: pd.DataFrame,
                      cluster_num: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign the cluster label to the mobility metrics dataframe
    :param mobility_before: the metric dataframe based on the mobility behavior before Covid19
    :param mobility_during: the metric dataframe based on the mobility behavior during Covid19
    :param cluster_num: the number of clusters we use for cluster analysis
    :return: the metric dataframe with cluster labels during and before the covid
    """
    assert 'user_id' in mobility_before, "The user id is missing"
    assert 'user_id' in mobility_during, "The user id is missing"
    # Determine the number of clusters
    data_colnames = ['mean_dist', 'std_dist', 'jump_dist', 'rg_val',
                     'mean_time', 'std_time', 'jump_time', 'tweet_num', 'tweet_time_span']
    data_array = mobility_before[data_colnames]
    cluster_inertia_dict = elbow_method(data_array, standardize_data=True)
    plot_inertias(cluster_inertia_dict, highest_cluster_num=15)
    # Cluster the Twitter users based on given metrics
    mobility_data_with_clusters_before = mobility_before.copy()
    mobility_data_with_clusters_during = mobility_during.copy()
    user_set_before, user_set_during = set(mobility_data_with_clusters_before['user_id']), \
                                       set(mobility_data_with_clusters_during['user_id'])
    assert len(set.symmetric_difference(user_set_before, user_set_during)) == 0, \
        "The user set does not match!"
    # Use the features before covid to cluster the users
    data_colnames = ['mean_dist', 'std_dist', 'jump_dist', 'rg_val',
                     'mean_time', 'std_time', 'jump_time', 'tweet_num', 'tweet_time_span']
    data_array = mobility_before[data_colnames]
    user_clusters = cluster_users(data_array=data_array, cluster_num=cluster_num)
    mobility_data_with_clusters_before['cluster_before'] = user_clusters
    # Assign the cluster label to the mobility dataframe during Covid
    user_cluster_info = mobility_data_with_clusters_before[['user_id', 'cluster_before']]
    merged_dataframe = pd.merge(left=mobility_data_with_clusters_during,
                                right=user_cluster_info, on='user_id')
    return merged_dataframe, mobility_data_with_clusters_before


if __name__ == '__main__':
    mobility_before_covid = pd.read_csv(os.path.join(records_path, 'user_mobility_before_covid.csv'),
                                        index_col=0, encoding='utf-8')
    mobility_during_covid = pd.read_csv(os.path.join(records_path, 'user_mobility_during_covid.csv'),
                                        index_col=0, encoding='utf-8')
    user_with_clusters_during, user_with_clusters_before = main_user_cluster(
        mobility_before=mobility_before_covid, cluster_num=7,
        mobility_during=mobility_during_covid)
    user_with_clusters_during.to_csv(os.path.join(records_path, 'user_mobility_with_clusters_during.csv'),
                                            encoding='utf-8')
    user_with_clusters_before.to_csv(os.path.join(records_path, 'user_mobility_with_clusters_before.csv'),
                                     encoding='utf-8')