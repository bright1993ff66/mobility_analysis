# -*- coding: utf-8 -*-
# @Time    : 2022/10/13 16:27
# @Author  : Haoliang Chang
import os
import time
import numpy as np
import pandas as pd

from data_paths import tweet_combined_path, records_path
from mobility_behavior.user_features import radius_of_gyration, distance_successive_points
from mobility_behavior.user_features import time_successive_points
from utils import create_geodataframe_from_csv, column_type_dict_filtered


"""
We define the mobility behavior in social media as a combination of footprints and activities
presented on the social media platform.
Internal properties (cluster the Twitter users):
    - distance (4 features)： mean, std, jump exists (10km) of successive distances, radius of gyration
    - time (3 features): mean, std, jump exists (1 month) of successive time gaps
    - tweet properties (2 features): number of tweets and tweet time span
External properties (Profile the Twitter users):
    - POI entropy
    - Activities (By POI class and text analysis)
"""


def count_tweet_time_span(tweet_dataframe: pd.DataFrame, user_ids: set) -> pd.DataFrame:
    """
    Count the number of tweets and time span of each user
    :param tweet_dataframe: a tweet dataframe
    :param user_ids: a set containing the ids of users who posted tweets both before
    and during the pandemic
    :return: a result dataframe saving the time span and tweet count of each user
    """
    user_list, time_span_list, tweet_count_list = [], [], []
    result_dataframe = pd.DataFrame()
    for user in user_ids:
        user_list.append(user)
        select_dataframe = tweet_dataframe.loc[tweet_dataframe['user_id_str'] == user]
        tweet_count_list.append(len(set(select_dataframe['id_str'])))
        try:
            time_span_days = (list(select_dataframe['hk_time'])[-1] -
                              list(select_dataframe['hk_time'])[0]).days
        except IndexError:
            time_span_days = 0
        time_span_list.append(time_span_days)
    result_dataframe['user_ids'] = user_list
    result_dataframe['time_span'] = time_span_list
    result_dataframe['tweet_count'] = tweet_count_list
    percent_less_seven = result_dataframe.loc[result_dataframe['time_span'] <= 7].shape[0]/len(user_list)
    percent_less_thirty = result_dataframe.loc[result_dataframe['time_span'] <= 30].shape[0]/len(user_list)
    print('Percent less than 7: {}; less than 30: {}'.format(round(percent_less_seven, 2),
                                                             round(percent_less_thirty, 2)))
    return result_dataframe


def compute_radius_gyration_users(tweet_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the radius of gyration for each user
    :param tweet_dataframe: the tweet dataframe
    :return: a dataframe containing the radius of gyration values for each user
    """
    assert 'user_id_str' in tweet_dataframe, \
        "The tweet dataframe should contain the user id information"
    tweet_dataframe_geo = create_geodataframe_from_csv(tweet_dataframe,
                                                       source_crs=4326, target_crs=4326,
                                                       accurate_pos=False)
    user_set = set(tweet_dataframe_geo['user_id_str'])
    rg_dataframe = pd.DataFrame()
    user_list, rg_list = [], []
    for user in user_set:
        user_list.append(user)
        user_dataframe = tweet_dataframe_geo.loc[tweet_dataframe_geo['user_id_str'] == user]
        rg_val = radius_of_gyration(point_data=user_dataframe)
        rg_list.append(rg_val)
    rg_dataframe['user_id'] = user_list
    rg_dataframe['rg_val'] = rg_list
    return rg_dataframe


def compute_successive_distances(tweet_dataframe: pd.DataFrame) -> dict:
    """
    Compute the successive distance between footprints of each user
    :param tweet_dataframe: the tweet dataframe
    :return: a dictionary saving the successive distances of each user
    """
    assert 'user_id_str' in tweet_dataframe, \
        "The tweet dataframe should contain the user id information"
    tweet_dataframe_geo = create_geodataframe_from_csv(tweet_dataframe,
                                                       source_crs=4326, target_crs=4316,
                                                       accurate_pos=False)
    user_set = set(tweet_dataframe_geo['user_id_str'])
    user_distance_dict = dict()
    for user in user_set:
        user_dataframe = tweet_dataframe_geo.loc[tweet_dataframe_geo['user_id_str'] == user]
        distance_array = distance_successive_points(point_data=user_dataframe)
        user_distance_dict[user] = distance_array
    return user_distance_dict


def compute_successive_time(tweet_dataframe: pd.DataFrame) -> dict:
    """
    Compute the successive time gaps between geocoded tweets of each user
    :param tweet_dataframe: the tweet dataframe
    :return: a dictionary saving the successive time gaps of each user
    """
    assert 'user_id_str' in tweet_dataframe, \
        "The tweet dataframe should contain the user id information"
    tweet_dataframe_geo = create_geodataframe_from_csv(tweet_dataframe,
                                                       source_crs=4326, target_crs=4316,
                                                       accurate_pos=False)
    user_set = set(tweet_dataframe_geo['user_id_str'])
    user_time_dict = dict()
    for user in user_set:
        user_dataframe = tweet_dataframe_geo.loc[tweet_dataframe_geo['user_id_str'] == user]
        time_array = time_successive_points(point_data=user_dataframe)
        user_time_dict[user] = time_array
    return user_time_dict


def main_successive_distance_time() -> None:
    """
    Main function to compute the radius of gyration, successive distances,
    and successive temporal gaps for each user
    :return: None. The results are saved to local directory
    """
    # Load the tweets posted before and during the pandemic
    start_time = time.time()
    print('Load the data...')
    tweet_before = pd.read_csv(os.path.join(tweet_combined_path, 'hk_common_2018_translated.csv'),
                               index_col=0, encoding='utf-8')
    tweet_during = pd.read_csv(os.path.join(tweet_combined_path, 'hk_common_covid_translated.csv'),
                               index_col=0, encoding='utf-8')
    print('Done! Compute the metrics....')
    # Compute the mobility metrics
    print('For radius of gyration...')
    rg_before = compute_radius_gyration_users(tweet_before)
    rg_during = compute_radius_gyration_users(tweet_during)
    print('For successive distances...')
    successive_dist_before = compute_successive_distances(tweet_before)
    successive_dist_during = compute_successive_distances(tweet_during)
    print('For successive temporal gaps...')
    successive_time_before = compute_successive_time(tweet_before)
    successive_time_during = compute_successive_time(tweet_during)
    print('Done! Save the results...')
    # Save the metrics to local directory
    rg_before.to_csv(os.path.join(records_path, 'rog_vals_2018.csv'), encoding='utf-8')
    rg_during.to_csv(os.path.join(records_path, 'rog_vals_covid.csv'), encoding='utf-8')
    np.save(os.path.join(records_path, 'distance_vals_2018_dict.npy'), successive_dist_before)
    np.save(os.path.join(records_path, 'distance_vals_covid_dict.npy'), successive_dist_during)
    np.save(os.path.join(records_path, 'time_gaps_2018_dict.npy'), successive_time_before)
    np.save(os.path.join(records_path, 'time_gaps_covid_dict.npy'), successive_time_during)
    end_time = time.time()
    print('Total time: {} min'.format(str(round((end_time-start_time)/60, 2))))


if __name__ == '__main__':
    main_successive_distance_time()


