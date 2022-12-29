"""
Find the tweets posted by users who ever posted tweets
both before and during the Covid19 pandemic
"""
import os
import numpy as np
import pandas as pd
from data_paths import tweet_combined_path
from utils import number_of_tweet_users
from utils import column_type_dict_filtered


def find_common_users(before_df: pd.DataFrame, during_df: pd.DataFrame) -> set:
    """
    Find the ids of Twitter users who posted tweets before and during the
    Covid19 pandemic
    :param before_df: a dataframe containing the tweets posted before pandemic
    :param during_df: a dataframe containing the tweets posted during pandemic
    :return a set containing the ids of users who ever posted tweets both before
    and during the Covid
    """
    before_users = set(before_df['user_id_str'])
    after_users = set(during_df['user_id_str'])
    common_users = set.intersection(before_users, after_users)
    print('One Common user: {}'.format(common_users.pop()))
    return common_users


def main_find_tweets_common_users():
    """
    Main function to find the tweets posted by common users
    :return: None. The tweets are saved to local directory
    """
    hk_2018_filtered_transformed = pd.read_csv(os.path.join(tweet_combined_path,
                                                            'hk_tweets_2018_filtered.csv'),
                                               encoding='utf-8', index_col=0,
                                               dtype=column_type_dict_filtered)
    hk_current_filtered_transformed = pd.read_csv(os.path.join(tweet_combined_path,
                                                               'hk_tweets_filtered.csv'),
                                                  encoding='utf-8', index_col=0,
                                                  dtype=column_type_dict_filtered)
    common_users_before_during_covid = find_common_users(before_df=hk_2018_filtered_transformed,
                                                         during_df=hk_current_filtered_transformed)
    np.save(os.path.join(tweet_combined_path, 'common_users.npy'),
            common_users_before_during_covid)
    hk_2018_from_common = \
        hk_2018_filtered_transformed.loc[hk_2018_filtered_transformed['user_id_str'].isin(
            common_users_before_during_covid)].reset_index(drop=True)
    hk_current_from_common = \
        hk_current_filtered_transformed.loc[hk_current_filtered_transformed['user_id_str'].isin(
            common_users_before_during_covid)].reset_index(drop=True)
    print('For the tweets posted before the Covid19 pandemic...')
    number_of_tweet_users(hk_2018_from_common, print_value=True)
    print('For the tweets posted during the Covid19 pandemic...')
    number_of_tweet_users(hk_current_from_common, print_value=True)
    hk_2018_from_common.to_csv(os.path.join(tweet_combined_path,
                                            'hk_tweets_2018_from_common_users.csv'),
                               encoding='utf-8')
    hk_current_from_common.to_csv(os.path.join(tweet_combined_path,
                                               'hk_tweets_current_from_common_users.csv'),
                                  encoding='utf-8')


if __name__ == '__main__':
    main_find_tweets_common_users()
