# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 17:44
# @Author  : Haoliang Chang
import os
from typing import Tuple
import numpy as np
import pandas as pd
from geopy.distance import geodesic

import geopandas as gpd
from data_paths import records_path, tweet_combined_path
from utils import hong_kong_epsg, transform_datetime_string_to_datetime
from utils import compute_time_span, prepare_tweet_data


def geodesic_distance(origin: tuple, destination: tuple) -> float:
    """
    Compute the geodesic distance between origin and destination (in kilometers)
    the coordinate system should be (latitude, longitude) in WGS84
    :param origin: the origin of trip
    :param destination: the destination of trip
    :return: the geodesic distance
    """
    return geodesic(origin, destination).km


def distance_successive_points(
        point_data: gpd.geodataframe.GeoDataFrame) -> np.ndarray:
    """
    Compute the geodesic distance of successive coordinates
    :param point_data: the geopandas dataframe saving the geocoded tweets posted by one user
    :return: the mean and standard deviation of distances of successive coordinates
    """
    # Transform the coordinate system of shapefile
    if point_data.crs.to_epsg() != 4326:
        point_data_transform = point_data.to_crs(epsg=4326)
    else:
        point_data_transform = point_data.copy()
    # Compute the successive distances
    num_points = point_data_transform.shape[0]
    distance_list = []
    first_pointer, second_pointer = 0, 1  # Create pointer to compute distance
    while second_pointer <= num_points - 1:
        first_point = point_data_transform.iloc[first_pointer]
        second_point = point_data_transform.iloc[second_pointer]
        distance = geodesic_distance(
            origin=(first_point['geometry'].y, first_point['geometry'].x),
            destination=(second_point['geometry'].y, second_point['geometry'].x))
        distance_list.append(distance)
        first_pointer += 1
        second_pointer += 1
    return np.array(distance_list)


def time_successive_points(
        point_data: gpd.geodataframe.GeoDataFrame) -> np.ndarray:
    """
    Compute the time between successive coordinates
    :param point_data: the geopandas dataframe saving the geocoded tweets posted by one user
    :return: the mean and standard deviation of time between successive coordinates
    """
    # Configure the time column
    assert 'hk_time' in point_data, "The point data should have a column named 'hk_time'"
    if isinstance(list(point_data['hk_time'])[0], str):
        point_data['hk_time'] = point_data.apply(
            lambda row: transform_datetime_string_to_datetime(row['hk_time']), axis=1)
    # Compute the successive distances
    point_data = point_data.reset_index(drop=True)
    num_points = point_data.shape[0]
    time_list = []
    first_pointer, second_pointer = 0, 1  # Create pointer to compute distance
    while second_pointer <= num_points - 1:
        first_point = point_data.iloc[first_pointer]
        second_point = point_data.iloc[second_pointer]
        # Compute the days between the successive geocoded tweets
        time_gap = (second_point['hk_time'] - first_point['hk_time']).days
        time_list.append(time_gap)
        first_pointer += 1
        second_pointer += 1
    return np.array(time_list)


def radius_of_gyration(point_data: gpd.geodataframe.GeoDataFrame) -> float:
    """
    Compute the radius of gyration based on geodesic distance
    ref: https://doi.org/10.1016/j.trc.2018.09.006
    :param point_data: the point shapefile
    :return: the radius of gyration
    """
    # Transform the coordinate system of shapefile
    # The coordinate system of point_data_transform is 2326
    if point_data.crs.to_epsg() == 4326:
        point_data_transform = point_data.to_crs(epsg=hong_kong_epsg)
        point_data_4326 = point_data.copy()
    elif point_data.crs.to_epsg() == hong_kong_epsg:
        point_data_transform = point_data.copy()
        point_data_4326 = point_data_transform.to_crs(epsg=4326)
    else:
        raise ValueError(
            'The coordinate system is not set properly. Should be {}'.format(hong_kong_epsg))
    # Get the center point given the distribution of point features
    # To compute the centroid, we should project the shapefile to the project coordinate system
    center_point = point_data_transform.dissolve().centroid
    center_point_4326 = center_point.to_crs(epsg=4326)
    center_point_pos = (center_point_4326.y[0], center_point_4326.x[0])
    # Compute the radius of gyration
    # To compute the geodesic distance, the points should be set in the coordinate system 4326
    radius_gyration_sum = 0
    num_points = point_data_4326.shape[0]
    for index, row in point_data_4326.iterrows():
        # geometry.y: latitude; geometry.x: longitude
        tweet_point = (row['geometry'].y, row['geometry'].x)
        radius_gyration_sum += np.power(geodesic_distance(origin=tweet_point,
                                                          destination=center_point_pos), 2)
    return np.sqrt(radius_gyration_sum / num_points)


def jump_exists(values_array: np.ndarray,
                value_threshold: float = 10) -> int:
    """
    Compute whether the jump exists given a threshold value
    :param values_array: the numpy array saving the successive distance or time gaps
    :param value_threshold: the threshold to determine the jump
    :return: 1 means if there is a jump and 0 otherwise
    """
    if np.max(values_array) >= value_threshold:
        return 1
    else:
        return 0


def mean_std_jump_distances(successive_distances: np.ndarray) -> Tuple[float, float, int]:
    """
    :param successive_distances: a numpy array saving the successive distance
    of a Twitter user
    :return: the mean, standardized distance, and whether the jump exists
    """
    if len(successive_distances) > 0:  # Compute the mean, std, and jump exists
        mean_distance = np.mean(successive_distances)
        std_distance = np.std(successive_distances)
        jump_exist = jump_exists(successive_distances, value_threshold=10)
    else:  # If only one tweet was posted
        mean_distance, std_distance, jump_exist = 0, 0, 0
    return mean_distance, std_distance, jump_exist


def mean_std_jump_time(successive_time_gaps: np.ndarray) -> Tuple[float, float, int]:
    """
    :param successive_time_gaps: a numpy array saving the successive time gaps of
    a Twitter user
    :return: the mean, standardized distance, and whether the jump exists
    """
    if len(successive_time_gaps) > 0:  # Compute the mean, std, and jump exists
        mean_distance = np.mean(successive_time_gaps)
        std_distance = np.std(successive_time_gaps)
        jump_exist = jump_exists(successive_time_gaps, value_threshold=30)
    else:  # If only one tweet was posted
        mean_distance, std_distance, jump_exist = 0, 0, 0
    return mean_distance, std_distance, jump_exist


def construct_user_mobility_data(distance_dict: dict, time_dict: dict,
                                 rog_vals_dataframe: pd.DataFrame,
                                 tweet_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a pandas dataframe saving the mobility metrics of all the users
    :param distance_dict: a python dictionary saving the successive distances of
    Twitter users. The distance dict can be calculated by compute_successive_distances in
    compute_mobility_behavior model
    :param time_dict: a python dictionary saving the successive time gaps of
    Twitter users. The time gap dict can be calculated by compute_successive_time in
    compute_mobility_behavior model
    :param rog_vals_dataframe: a pandas dataframe saving the radius gyration of users
    :param tweet_dataframe: the whole tweet dataframe
    :return: a pandas dataframe saving the mobility metrics of Twitter users
    """
    processed_tweets = prepare_tweet_data(tweet_dataframe_csv=tweet_dataframe)
    result_dataframe = pd.DataFrame()
    user_list = []
    mean_dist_list, std_dist_list, jump_dist_list = [], [], []
    mean_time_list, std_time_list, jump_time_list = [], [], []
    tweet_num_list, tweet_time_span_list = [], []
    for user_id in distance_dict:
        user_list.append(user_id)
        selected_tweets = processed_tweets.loc[processed_tweets['user_id_str'] == user_id]
        tweet_num_list.append(len(set(selected_tweets['id_str'])))
        tweet_time_span_list.append(compute_time_span(tweet_dataframe=selected_tweets))
        distances = distance_dict[user_id]
        time_gaps = time_dict[user_id]
        mean_dist, std_dist, jump_dist = mean_std_jump_distances(distances)
        mean_time, std_time, jump_time = mean_std_jump_distances(time_gaps)
        mean_dist_list.append(mean_dist)
        std_dist_list.append(std_dist)
        jump_dist_list.append(jump_dist)
        mean_time_list.append(mean_time)
        std_time_list.append(std_time)
        jump_time_list.append(jump_time)
    result_dataframe['user_id'] = user_list
    result_dataframe['tweet_num'] = tweet_num_list
    result_dataframe['tweet_time_span'] = tweet_time_span_list
    result_dataframe['mean_dist'] = mean_dist_list
    result_dataframe['std_dist'] = std_dist_list
    result_dataframe['jump_dist'] = jump_dist_list
    result_dataframe['mean_time'] = mean_time_list
    result_dataframe['std_time'] = std_time_list
    result_dataframe['jump_time'] = jump_time_list
    assert set(result_dataframe['user_id']) == set(rog_vals_dataframe['user_id']), \
        "The user ids of radius of gyration values and other metrics differ"
    combined_dataframe = pd.merge(left=result_dataframe, right=rog_vals_dataframe,
                                  on='user_id')
    return combined_dataframe


def main_user_features() -> None:
    """
    Main function to compute the metrics for each user and store in dataframes
    :return: None. The results are saved to local directory
    """
    # Load the distance-based and time-based metrics
    distance_2018_dict = np.load(os.path.join(records_path, 'distance_vals_2018_dict.npy'),
                                 allow_pickle=True).item()
    distance_covid_dict = np.load(os.path.join(records_path, 'distance_vals_covid_dict.npy'),
                                  allow_pickle=True).item()
    time_gaps_2018_dict = np.load(os.path.join(records_path, 'time_gaps_2018_dict.npy'),
                                  allow_pickle=True).item()
    time_gaps_covid_dict = np.load(os.path.join(records_path, 'time_gaps_covid_dict.npy'),
                                   allow_pickle=True).item()
    rog_vals = pd.read_csv(os.path.join(records_path, 'rog_vals_2018.csv'), index_col=0)
    rog_vals_covid = pd.read_csv(os.path.join(records_path, 'rog_vals_covid.csv'), index_col=0)
    hk_tweets_prev = pd.read_csv(os.path.join(tweet_combined_path,
                                              'hk_common_2018_translated.csv'), index_col=0)
    hk_tweets_covid = pd.read_csv(os.path.join(tweet_combined_path,
                                               'hk_common_covid_translated.csv'), index_col=0)
    # Construct the mobility dataframes
    user_mobility_before_covid = construct_user_mobility_data(rog_vals_dataframe=rog_vals,
                                                              distance_dict=distance_2018_dict,
                                                              time_dict=time_gaps_2018_dict,
                                                              tweet_dataframe=hk_tweets_prev)
    user_mobility_during_covid = construct_user_mobility_data(
        rog_vals_dataframe=rog_vals_covid,
        distance_dict=distance_covid_dict,
        time_dict=time_gaps_covid_dict,
        tweet_dataframe=hk_tweets_covid)
    user_mobility_before_covid.to_csv(os.path.join(records_path, 'user_mobility_before_covid.csv'),
                                      encoding='utf-8')
    user_mobility_during_covid.to_csv(os.path.join(records_path, 'user_mobility_during_covid.csv'),
                                      encoding='utf-8')


if __name__ == '__main__':
    main_user_features()
