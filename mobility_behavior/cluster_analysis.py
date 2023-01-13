# -*- coding: utf-8 -*-
# @Time    : 2023/1/11 16:57
# @Author  : Haoliang Chang
# basics
import os
from typing import Tuple
import numpy as np
import pandas as pd

# clustering modules
from sklearn.cluster import KMeans, DBSCAN

# spatial analysis
import geopandas as gpd

# Load other paths and utilities
from utils import random_seed, hong_kong_epsg
from visualizations import plot_inertias
from visualizations import before_color_code, during_color_code
from data_paths import records_path, shapefile_path


def elbow_method(data_arr: np.array, max_cluster_num: int = 20) -> dict:
    """
    Use elbow method to select the best cluster number. For reference, please check:
    https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    :param data_arr: a numpy array saving the data we want to cluster
    :param max_cluster_num: the maximum number of clusters for consideration
    :return: one dictionary saving the inertia values
    """
    inertias = []
    mapping = {}

    K = range(3, max_cluster_num+1)  # Specify the number of clusters we want to try (3 to 20)

    for k in K:
        # Building and fitting the model
        print('Checking the cluster num: {}'.format(k))
        kmeanModel = KMeans(n_clusters=k, max_iter=10, random_state=random_seed)
        kmeanModel.fit(data_arr)

        # # Get the cluster centers
        # centroids = kmeanModel.cluster_centers_

        # Store the inertias
        inertia = kmeanModel.inertia_  # Sum of squared distances of samples to their closest cluster center
        inertias.append(inertia)

        # Create the result dictionary
        mapping[k] = inertia

    return mapping


def kmeans_cluster_results(data_array: np.ndarray,
                           cluster_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the cluster center and cluster labels based on KMeans clustering
    :param data_array: a numpy array saving the locations of geotagged tweets
    :param cluster_num: the cluster number selected by elbow method
    :return: two numpy arrays: cluster centers and cluster labels
    """
    kmeans = KMeans(n_clusters=cluster_num, random_state=random_seed, max_iter=10).fit(
        data_array)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    return cluster_centers, cluster_labels


def main_cluster_analysis(cluster_number_dict: dict):
    """
    Main function to conduct the cluster analysis
        - cluster 0 before: 8; cluster 0 during: 6
        - cluster 1 before: 6; cluster 1 during: 7
        - cluster 2 before: 8; cluster 2 during: 9
        - cluster 3 before: 9; cluster 3 during: 9
        - cluster 6 before: 7; cluster 6 during: 7
    :param cluster_number_dict: a dictionary saving the cluster number before and during
    the pandemic
    :return:
    """
    # Load the locations of geotagged tweets posted by each group of Twitter user
    cluster_labels = [str(cluster_num) for cluster_num in range(0, 7)]
    tweet_path = os.path.join(records_path, 'user_cluster_tweets', 'tweets_csv')
    lat_colname, lon_colname = 'place_lat', 'place_lon'
    for user_cluster in cluster_labels:
        print('***Clustering the tweet locations of user cluster {}***'.format(user_cluster))
        tweet_before = pd.read_csv(
            os.path.join(tweet_path, 'before_tweets_cluster_{}.csv'.format(user_cluster)),
            index_col=0, encoding='utf-8')
        tweet_during = pd.read_csv(
            os.path.join(tweet_path, 'during_tweets_cluster_{}.csv'.format(user_cluster)),
            index_col=0, encoding='utf-8')
        if len(set(tweet_during['user_id_str'])) >= 50:
            print('We have {} users'.format(len(set(tweet_during['user_id_str']))))
            locs_before = tweet_before[[lat_colname, lon_colname]].to_numpy()
            locs_during = tweet_during[[lat_colname, lon_colname]].to_numpy()
            # # Choose the number of clusters for KMeans clustering
            # elbow_dict_before = elbow_method(data_arr=locs_before)
            # elbow_dict_during = elbow_method(data_arr=locs_during)
            # plot_inertias(inertia_dict=elbow_dict_before,
            #               save_filename='before_cluster_{}_elbow.png'.format(user_cluster),
            #               line_color=before_color_code)
            # plot_inertias(inertia_dict=elbow_dict_during,
            #               save_filename='during_cluster_{}_elbow.png'.format(user_cluster),
            #               line_color=during_color_code)

            # Use KMeans to cluster the locations
            cluster_centers_before, cluster_labels_before = kmeans_cluster_results(
                data_array=locs_before, cluster_num=cluster_number_dict[int(user_cluster)][0])
            cluster_centers_during, cluster_labels_during = kmeans_cluster_results(
                data_array=locs_during, cluster_num=cluster_number_dict[int(user_cluster)][1])
            print(cluster_centers_before)
            tweet_before['cluster_label'] = cluster_labels_before
            tweet_during['cluster_label'] = cluster_labels_during
            cluster_centers_before_dataframe = pd.DataFrame(cluster_centers_before,
                                                            columns=['lat', 'lon'])
            cluster_centers_during_dataframe = pd.DataFrame(cluster_centers_during,
                                                            columns=['lat', 'lon'])
            cluster_centers_geo_before = gpd.GeoDataFrame(cluster_centers_before_dataframe,
                                                          geometry=gpd.points_from_xy(
                                                              cluster_centers_before_dataframe.lon,
                                                   cluster_centers_before_dataframe.lat)).set_crs(
                epsg=4326, inplace=True).to_crs(hong_kong_epsg)
            cluster_centers_geo_during = gpd.GeoDataFrame(cluster_centers_during_dataframe,
                                                          geometry=gpd.points_from_xy(
                                                              cluster_centers_during_dataframe.lon,
                                                              cluster_centers_during_dataframe.lat)).set_crs(
                epsg=4326, inplace=True).to_crs(hong_kong_epsg)
            # Save the created shapefiles and tweet datasets to local directories
            cluster_centers_geo_before.to_file(os.path.join(
                shapefile_path, 'kmeans_loc_center_before_{}.shp'.format(user_cluster)), encoding='utf-8')
            cluster_centers_geo_during.to_file(os.path.join(
                shapefile_path, 'kmeans_loc_center_during_{}.shp'.format(user_cluster)), encoding='utf-8')
            tweet_before.to_csv(os.path.join(
                records_path, '{}_user_with_loc_cluster_before.csv'.format(user_cluster)), encoding='utf-8')
            tweet_during.to_csv(os.path.join(
                records_path, '{}_user_with_loc_cluster_during.csv'.format(user_cluster)), encoding='utf-8')
        else:
            print('Cluster {} has less than 50 users. Ignore.'.format(user_cluster))


if __name__ == '__main__':
    cluster_num_dict = {0: [8, 6], 1: [6, 7], 2: [8, 9], 3: [9, 9], 6: [7, 7]}
    main_cluster_analysis(cluster_number_dict=cluster_num_dict)
