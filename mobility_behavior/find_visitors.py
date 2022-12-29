# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 17:27
# @Author  : Haoliang Chang
import os
from typing import Tuple
import pandas as pd

from data_paths import tweet_combined_path
from utils import transform_datetime_string_to_datetime
from utils import general_info_of_tweet_dataset


def find_temporary_visitors(tweet_dataframe: pd.DataFrame,
                            time_span_limit: int = 7) -> set:
    """
    Find the temporary visitors: time span less than 7 days
    ref: https://doi.org/10.1016/j.tranpol.2022.03.011
    :param tweet_dataframe: a tweet dataframe saving the collected tweets
    :param time_span_limit: the limit of time span
    :return: a python set saving the id of temporary visitors
    """
    user_set = set(tweet_dataframe['user_id_str'])
    if isinstance(list(tweet_dataframe['hk_time'])[0], str):
        tweet_dataframe['hk_time'] = tweet_dataframe.apply(
            lambda row: transform_datetime_string_to_datetime(row['hk_time']), axis=1)
    tweet_dataframe_sorted = tweet_dataframe.sort_values(by='hk_time').reset_index(drop=True)
    temporary_visitor_set = set()
    for user in user_set:
        user_tweets = tweet_dataframe_sorted.loc[tweet_dataframe_sorted['user_id_str'] == user]
        start_time = list(user_tweets['hk_time'])[0]
        end_time = list(user_tweets['hk_time'])[-1]
        time_span = (end_time - start_time).days
        if time_span <= time_span_limit:
            temporary_visitor_set.add(user)
    return temporary_visitor_set


def find_international_tourists(tweet_dataframe: pd.DataFrame,
                                time_span_limit: int = 90,
                                active_days_limit: int = 30) -> set:
    """
    Find the international tourists
    ref: https://doi.org/10.1016/j.apgeog.2016.06.001
    :param tweet_dataframe: a tweet dataframe saving the collected tweets
    :param time_span_limit: the limit of time span
    :param active_days_limit: the limit of active days
    :return: a python set saving the id of international tourists
    """
    user_set = set(tweet_dataframe['user_id_str'])
    if isinstance(list(tweet_dataframe['hk_time'])[0], str):
        tweet_dataframe['hk_time'] = tweet_dataframe.apply(
            lambda row: transform_datetime_string_to_datetime(row['hk_time']), axis=1)
    tweet_dataframe_sorted = tweet_dataframe.sort_values(by='hk_time').reset_index(drop=True)
    international_tourists_set = set()
    for user in user_set:
        user_tweets = tweet_dataframe_sorted.loc[tweet_dataframe_sorted['user_id_str'] == user].copy()
        user_tweets['day_info'] = user_tweets.apply(
            lambda row: str(row['year']) + '_' + str(row['month']) + '_' + str(row['day']), axis=1)
        active_days = len(set(user_tweets['day_info']))
        start_time = list(user_tweets['hk_time'])[0]
        end_time = list(user_tweets['hk_time'])[-1]
        time_span = (end_time - start_time).days
        if (time_span <= time_span_limit) & (active_days <= active_days_limit):
            international_tourists_set.add(user)
    return international_tourists_set


def main(tweet_dataframe, print_info=True) -> Tuple[set, set]:
    """
    Generate the description of the tweet dataframe
    :param tweet_dataframe: a tweet dataframe
    :param print_info: whether print the information or not
    :return: None. print the general descriptions of tweets posted by temporary visitors
    and international tourists
    """
    # Find the user ids
    all_users = set(tweet_dataframe['user_id_str'])
    temporary_visitors = find_temporary_visitors(tweet_dataframe)
    international_tourists = find_international_tourists(tweet_dataframe)
    others_not_temporary_visitors = all_users - temporary_visitors
    other_not_international_tourists = all_users - international_tourists
    # Find the tweets
    visitor_tweets = tweet_dataframe.loc[tweet_dataframe['user_id_str'].isin(
        temporary_visitors)]
    tourists_tweets = tweet_dataframe.loc[tweet_dataframe['user_id_str'].isin(
        international_tourists)]
    other_not_visitor_tweets = tweet_dataframe.loc[tweet_dataframe['user_id_str'].isin(
        others_not_temporary_visitors)]
    other_not_tourists_tweets = tweet_dataframe.loc[tweet_dataframe['user_id_str'].isin(
        other_not_international_tourists)]
    if print_info:
        # Generate the tweet descriptions
        print('*'*10+' For temporary visitors '+'*'*10)
        general_info_of_tweet_dataset(df=visitor_tweets)
        print('*' * 10 + ' Users not temporary visitors ' + '*' * 10)
        general_info_of_tweet_dataset(df=other_not_visitor_tweets)
        print('*' * 10 + ' For international tourists ' + '*' * 10)
        general_info_of_tweet_dataset(df=tourists_tweets)
        print('*' * 10 + ' Users not international tourists ' + '*' * 10)
        general_info_of_tweet_dataset(df=other_not_tourists_tweets)
    else:
        # Return the ids of temporary visitors and international tourists
        print('Return the ids of temporary visitors and international tourists...')
    return temporary_visitors, international_tourists


if __name__ == '__main__':
    hk_tweets_2018_common = pd.read_csv(os.path.join(
        tweet_combined_path, 'hk_tweets_2018_from_common_users.csv'), index_col=0, encoding='utf-8')
    hk_tweets_covid_common = pd.read_csv(os.path.join(
        tweet_combined_path, 'hk_tweets_current_from_common_users.csv'), index_col=0, encoding='utf-8')
    _, international_visitors = main(tweet_dataframe=hk_tweets_2018_common, print_info=False)
    print('The number of international visitors: {}'.format(len(international_visitors)))
    before_covid_international_tweets = hk_tweets_2018_common.loc[
        hk_tweets_2018_common['user_id_str'].isin(international_visitors)]
    current_international_tweets = hk_tweets_covid_common.loc[
        hk_tweets_covid_common['user_id_str'].isin(international_visitors)]
    print('***For the tweets posted by international tourists before Covid...***')
    general_info_of_tweet_dataset(df=before_covid_international_tweets)
    print('***For the tweets posted by international tourists during Covid...***')
    general_info_of_tweet_dataset(df=current_international_tweets)
