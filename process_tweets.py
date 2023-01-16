"""
The process_tweets.py saves functions to combine some tweets
"""
# basics
import os
from collections import Counter
from typing import Tuple

# dataframe processing
import numpy as np
import pandas as pd

# Spatial analysis
import geopandas as gpd

# load utilities
from data_paths import tweet_path, tweet_combined_path, tweet_prev, records_path
from data_paths import shapefile_path
from utils import transform_string_time_to_datetime, hong_kong_epsg
from utils import combine_some_data, timezone_hongkong, get_time_attributes
from utils import column_type_dict_filtered, create_geodataframe_from_csv
from visualizations import plot_geo_points_over_polygons


def get_tweets_in_some_months(interested_months: list,
                              data_path: str = tweet_path,
                              save_path: str = tweet_combined_path,
                              save_csv=True) -> pd.DataFrame:
    """
    Get the tweets posted in some months
    :param interested_months: a list containing interested months
    :param data_path: the path saving the tweets
    :param save_path: the path used to save the combined data
    :param save_csv: save the combined file in the csv or not
    :return: a dataframe saving the tweets posted in interested
    months
    """
    combined_tweets = combine_some_data(path=data_path,
                                        sample_num=None)
    combined_tweets['hk_time'] = combined_tweets.apply(
        lambda row: transform_string_time_to_datetime(
            row['created_at'],
            target_time_zone=timezone_hongkong,
            convert_utc_time=True), axis=1)
    combined_tweets_sorted = combined_tweets.sort_values(by='hk_time')
    combined_tweets_with_time = get_time_attributes(combined_tweets_sorted)
    combined_tweets_select = combined_tweets_with_time.loc[
        combined_tweets_with_time['month'].isin(
            interested_months)].copy().reset_index(drop=True)
    if save_csv:
        combined_tweets_select.to_csv(
            os.path.join(save_path, 'hk_tweets_{}.csv'.format(
                interested_months)), encoding='utf-8')
    else:
        combined_tweets_select['hk_time'] = combined_tweets_select[
            'hk_time'].astype(str)
        combined_tweets_select.to_excel(
            os.path.join(save_path, 'hk_tweets_{}.xlsx'.format(
                interested_months)), engine='xlsxwriter')
    return combined_tweets_select


def get_combined_tweets(data_path: str = tweet_path,
                        save_path: str = tweet_combined_path,
                        save_csv: bool = True,
                        save_filename='hk_tweets_combined') -> pd.DataFrame:
    """
    Create a file containing all the tweets saved in a local path
    :param data_path: the path saving the tweets
    :param save_path: the path used to save the combined data
    :param save_csv: save the combined file in the csv or not
    :param save_filename: the filename of the created pandas dataframe
    :return: a dataframe containing the combined tweets
    months
    """
    combined_tweets = combine_some_data(path=data_path,
                                        sample_num=None,
                                        get_geocoded=False)
    combined_tweets['hk_time'] = combined_tweets.apply(
        lambda row: transform_string_time_to_datetime(
            row['created_at'],
            target_time_zone=timezone_hongkong,
            convert_utc_time=True), axis=1)
    combined_tweets_sorted = combined_tweets.sort_values(by='hk_time')
    combined_tweets_with_time = get_time_attributes(
        combined_tweets_sorted)
    time_list_check = ['created_at', 'hk_time', 'month', 'day',
                       'hour', 'minute']
    print(combined_tweets_with_time.sample(7)[time_list_check])
    if save_csv:
        combined_tweets_with_time.to_csv(
            os.path.join(save_path, '{}.csv'.format(save_filename)),
            encoding='utf-8')
    else:
        combined_tweets_with_time['hk_time'] = combined_tweets_with_time[
            'hk_time'].astype(str)
        combined_tweets_with_time.to_excel(
            os.path.join(save_path, '{}.xlsx'.format(save_filename)),
            engine='xlsxwriter')
    return combined_tweets_with_time


def find_tweets_posted_by_user_cluster() -> None:
    """
    Find the tweets posted by each cluster of Twitter users
    :return: The tweets posted by each user cluster are saved to local directory
    """
    # Load the tweet dataframes
    tweets_before = pd.read_csv(os.path.join(
        tweet_combined_path, 'hk_common_2018_translated.csv'),
        index_col=0, encoding='utf-8', dtype=column_type_dict_filtered)
    tweets_during = pd.read_csv(os.path.join(
        tweet_combined_path, 'hk_common_covid_translated.csv'),
        index_col=0, encoding='utf-8', dtype=column_type_dict_filtered)
    user_mobility_data = pd.read_csv(os.path.join(
        records_path, 'user_mobility_with_clusters_before.csv'), index_col=0,
        encoding='utf-8', dtype={'user_id': str, 'cluster_before': int})
    city_shape = gpd.read_file(os.path.join(shapefile_path, 'hk_tpu.shp'), encoding='utf-8')
    city_shape_project = city_shape.to_crs(epsg=hong_kong_epsg)
    # Get the tweets posted by each user cluster
    user_cluster_set = set(user_mobility_data['cluster_before'])
    print('User cluster counter: {}'.format(Counter(user_mobility_data['cluster_before'])))
    for cluster_label in user_cluster_set:
        print('*' * 20)
        print('Coping with the cluster: {}'.format(cluster_label))
        selected_mobility = user_mobility_data.loc[user_mobility_data['cluster_before'] == cluster_label]
        selected_user = set(selected_mobility['user_id'])
        print('Number of selected user: {}'.format(len(selected_user)))
        select_tweets_before = tweets_before.loc[tweets_before['user_id_str'].isin(selected_user)].reset_index(
            drop=True)
        select_tweets_during = tweets_during.loc[tweets_during['user_id_str'].isin(selected_user)].reset_index(
            drop=True)
        select_tweets_before_geo = create_geodataframe_from_csv(
            dataframe=select_tweets_before, source_crs=4326, target_crs=hong_kong_epsg,
            accurate_pos=False)
        select_tweets_during_geo = create_geodataframe_from_csv(
            dataframe=select_tweets_during, source_crs=4326, target_crs=hong_kong_epsg,
            accurate_pos=False)
        plot_geo_points_over_polygons(city_shape=city_shape_project,
                                      tweet_before_shape=select_tweets_before_geo,
                                      tweet_during_shape=select_tweets_during_geo,
                                      save_filename='tweet_geo_compare_cluster_{}.png'.format(cluster_label))
        print('Num of tweets posted before Covid: {}'.format(len(
            set(select_tweets_before_geo['id_str']))))
        print('Num of tweets posted during Covid: {}'.format(len(
            set(select_tweets_during_geo['id_str']))))
        select_tweets_before.to_csv(os.path.join(
            records_path, 'user_cluster_tweets', 'tweets_csv', 'before_tweets_cluster_{}.csv'.format(cluster_label)),
            encoding='utf-8')
        select_tweets_during.to_csv(os.path.join(
            records_path, 'user_cluster_tweets', 'tweets_csv', 'during_tweets_cluster_{}.csv'.format(cluster_label)),
            encoding='utf-8')
        select_tweets_before_geo.to_file(os.path.join(
            records_path, 'user_cluster_tweets', 'tweets_shapefile',
            'before_tweets_geo_cluster_{}.shp'.format(cluster_label)),
            encoding='utf-8')
        select_tweets_during_geo.to_file(os.path.join(
            records_path, 'user_cluster_tweets', 'tweets_shapefile',
            'during_tweets_geo_cluster_{}.shp'.format(cluster_label)),
            encoding='utf-8')
        print('*' * 20 + '\n')


def find_tweets_posted_by_same_users(prev_tweet_data: pd.DataFrame,
                                     covid_tweet_data: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find the tweets posted by users who ever posted tweets both in 2018 and
    after Covid19
    :param prev_tweet_data: the filtered tweet data posted in 2018
    :param covid_tweet_data: the filtered tweet data posted during Covid19
    :return:
    """
    assert 'user_id_str' in prev_tweet_data, "The dataframe should contain user id"
    assert 'user_id_str' in covid_tweet_data, "The dataframe should contain user id"
    # Get the user ids who ever posted tweets both in 2018 and after covid pandemic
    user_id_colname = 'user_id_str'
    set_prev = set(prev_tweet_data[user_id_colname])
    set_covid = set(covid_tweet_data[user_id_colname])
    intersect_set = set.intersection(set_prev, set_covid)
    np.save(os.path.join(tweet_combined_path, 'users_in_two_periods.npy'), intersect_set)
    # Return the tweet dataframe posted by common users
    tweet_data_prev_common_users = prev_tweet_data.loc[
        prev_tweet_data['user_id_str'].isin(intersect_set)]
    tweet_data_covid_common_users = covid_tweet_data.loc[
        covid_tweet_data['user_id_str'].isin(intersect_set)]
    tweet_data_prev_common_users.to_csv(os.path.join(
        tweet_combined_path, 'hk_tweets_2018_from_common_users.csv'), encoding='utf-8')
    tweet_data_covid_common_users.to_csv(os.path.join(
        tweet_combined_path, 'hk_tweets_current_from_common_users.csv'), encoding='utf-8')
    return tweet_data_prev_common_users, tweet_data_covid_common_users


if __name__ == '__main__':
    # print("Combine all the collected tweets in local")
    # print('Process the tweets posted in 2018...')
    # get_combined_tweets(data_path=tweet_prev, save_filename='hk_tweets_2018')
    # print('Process the tweets posted in the recent year...')
    # get_combined_tweets()
    # print('Find the tweets posted by the common users...')
    # prev_filter_data = pd.read_csv(os.path.join(tweet_combined_path,
    #                                             'hk_tweets_2018_filtered.csv'), index_col=0,
    #                                encoding='utf-8')
    # cur_filter_data = pd.read_csv(os.path.join(tweet_combined_path,
    #                                            'hk_tweets_filtered.csv'), index_col=0,
    #                               encoding='utf-8')
    # find_tweets_posted_by_same_users(prev_tweet_data=prev_filter_data,
    #                                  covid_tweet_data=cur_filter_data)
    # # Then use text_translate.py and text_translate_before.py to translate the tweets
    # Finally, find the tweets posted by each cluster of Twitter users
    print('Find the tweets posted by each Twitter user cluster...')
    find_tweets_posted_by_user_cluster()
