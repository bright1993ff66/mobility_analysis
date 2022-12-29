# basics
import pandas as pd
import os
import numpy as np
from collections import Counter
from datetime import datetime

# visualization
from matplotlib import pyplot as plt

# load the shapefiles
import geopandas as gpd
from utils import spatial_join, create_geodataframe_from_csv
from utils import general_info_of_tweet_dataset
from utils import transform_string_time_to_datetime
from utils import timezone_hongkong
from utils import write_to_excel
from utils import filter_rows_having_strings

# load the path for combined tweets
from data_paths import tweet_combined_path, shapefile_path

# Possible starting time and ending time
starting_time = datetime(2021, 6, 1, 0, 0, 0, tzinfo=timezone_hongkong)
ending_time = datetime(2100, 12, 19, 23, 59, 59, tzinfo=timezone_hongkong)
starting_time_2018 = datetime(2018, 1, 1, 0, 0, 0, tzinfo=timezone_hongkong)
ending_time_2018 = datetime(2018, 12, 31, 23, 59, 59, tzinfo=timezone_hongkong)


def count_user_tweet(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Count the users and the number of tweets they post.
    Bot accounts are likely to post many tweets in a long time
    :param dataframe: a tweet dataframe
    :return: a pandas dataframe saving the number of tweets posted by each user
    """
    assert 'user_id_str' in dataframe, "The dataframe must contain id info"
    user_set = set(dataframe['user_id_str'])
    user_list, tweet_count_list, tweet_pos_percent_list = [], [], []
    for user in user_set:
        data_select = dataframe.loc[dataframe['user_id_str'] == user].copy()
        if data_select.shape[0] > 0:
            user_list.append(user)
            data_select['pos'] = data_select.apply(
                lambda row: (row.lat, row.lon), axis=1)
            _, most_common_count = Counter(data_select['pos']).most_common()[0]
            tweet_count_list.append(data_select.shape[0])
            tweet_pos_percent_list.append(
                most_common_count / len(set(data_select['id_str'])))
    count_data = pd.DataFrame()
    count_data['user_id'] = user_list
    count_data['count'] = tweet_count_list
    count_data['loc_percent'] = tweet_pos_percent_list
    count_data_final = count_data.sort_values(
        by='count', ascending=False).reset_index(drop=True)
    return count_data_final


def get_bot_users(count_dataframe: pd.DataFrame, save_path: str,
                  save_filename: str) -> np.ndarray:
    """
    Get the user ids that are bot accounts. Some works for reference:
    https://www.mdpi.com/2078-2489/9/5/102/htm
    https://www.sciencedirect.com/science/article/pii/S0001457517302269
    :param count_dataframe: the pandas dataframe counting the tweet count and loc percent
    :param save_path: the path to the save the bot user ids
    :param save_filename: the name of the saved file
    :return: None. The bot ids are saved to the local directory
    """
    assert 'count' in count_dataframe, 'The count dataframe should have a column named count'
    assert 'loc_percent' in count_dataframe, 'The count dataframe should have a column named loc_percent'

    tweet_count_mean = np.mean(count_dataframe['count'])
    tweet_count_std = np.std(count_dataframe['count'])
    # Compute the threshold
    threshold = tweet_count_mean + 2 * tweet_count_std
    # Create the decision to find the bot accounts
    # - number of tweets exceeding the threshold
    # - location percentage bigger than 60%
    decision = (count_dataframe['count'] > threshold) & (
            count_dataframe['loc_percent'] > 0.6)
    bot_count_dataframe = count_dataframe[decision]
    bot_ids = np.array(list(set(bot_count_dataframe['user_id'])))
    print('We have got {} bots'.format(len(bot_ids)))
    print('They posted {} tweets'.format(sum(bot_count_dataframe['count'])))
    np.save(os.path.join(save_path, save_filename), bot_ids)
    return bot_ids


def main_func(filename: str, path: str = tweet_combined_path,
              interested_keywords: list = [],
              bot_savefilename: str = 'hk_bots.npy',
              accurate_position: bool = True,
              save_filename: str = 'hk_tweets_filtered.csv',
              start_time=starting_time, end_time=ending_time) -> pd.DataFrame:
    """
    Main function to find the tweets we can use in HK
    Filter steps:
        - Only consider the geocoded tweets
        - Find the tweets posted within HK boundary
        - Remove bots
    To be updated:
        - Which language should we use? zh, en, ja, in, tl?
    :param filename: the filename of the considered tweets
    :param path: the path loading and saving the interested file
    :param interested_keywords: a list containing the interested keywords
    :param accurate_position: whether use the accurate position (lat & lon)
    shared by the twitter users
    :param bot_savefilename: the filename of the saved file containing bot accounts
    :param save_filename: the filename of the saved file
    :param start_time: the starting time for the filtered tweets
    :param end_time: the ending time for the filtered tweets
    :return: None. The filtered tweets are saved to path
    """
    # Load the tweets and shapefile
    tweets = pd.read_csv(os.path.join(path, filename), index_col=0,
                         encoding='utf-8')
    geocoded_geo_data = create_geodataframe_from_csv(dataframe=tweets, source_crs=4326,
                                                     target_crs=4326,
                                                     accurate_pos=accurate_position)
    hk_shape = gpd.read_file(os.path.join(shapefile_path, 'hk_tpu.shp'),
                             encoding='utf-8')
    tweets_in_hk = spatial_join(point_gdf=geocoded_geo_data, shape_area=hk_shape)
    print("We have collected {} geocoded tweets posted in HK".format(
        len(set(tweets_in_hk['id_str']))))
    # Create the count data and find the bot accounts
    count_dataframe = count_user_tweet(dataframe=tweets_in_hk)
    hk_bots = get_bot_users(count_dataframe=count_dataframe, save_path=path,
                            save_filename=bot_savefilename)
    # hk_bots = np.load(os.path.join(path, bot_savefilename),
    #                  allow_pickle=True).tolist()
    final_tweets = tweets_in_hk.loc[~tweets_in_hk['user_id_str'].isin(hk_bots)]
    final_tweets['hk_time'] = final_tweets.apply(
        lambda row: transform_string_time_to_datetime(
            row['created_at'], timezone_hongkong), axis=1)
    time_list = ['created_at', 'hk_time', 'month', 'day',
                 'hour', 'minute']
    print(final_tweets.sample(7)[time_list])
    # Only consider the tweets posted in a time range
    time_mask_start = (final_tweets['hk_time'] >= start_time)
    time_mask_end = (final_tweets['hk_time'] <= end_time)
    final_tweets_in_time = final_tweets.loc[time_mask_start & time_mask_end]
    final_tweets_copy = final_tweets_in_time.copy()
    # Remove verified accounts
    if isinstance(list(final_tweets_copy['verified'])[0], str):
        final_tweets_without_verified = final_tweets_copy.loc[
            final_tweets_copy['verified'] == 'False']
    else:
        final_tweets_without_verified = final_tweets_copy.loc[
            final_tweets_copy['verified'] == False]
    # Filter the dataframe if keywords are given
    if interested_keywords:
        final_tweets_without_verified = filter_rows_having_strings(
            final_tweets_without_verified, interested_keywords, 'text')
    final_tweets_sorted = final_tweets_without_verified.sort_values(
        by='hk_time').reset_index(drop=True)
    general_info_of_tweet_dataset(final_tweets_sorted)
    final_tweets_sorted.to_csv(os.path.join(path, save_filename), encoding='utf-8')
    return final_tweets_sorted


if __name__ == '__main__':
    print('Process all the collected tweets')
    main_func(filename='hk_tweets_2018.csv',
              bot_savefilename='hk_bots_2018.npy',
              save_filename='hk_tweets_2018_filtered.csv',
              start_time=starting_time_2018, end_time=ending_time_2018)
    print('Find the tweets discussing some topics')
#    mtr_keywords = ['MTR', 'metro', 'subway', 'bus', 'minibus', '小巴', '紅VAN',
#               '綠Van', '港鐵', '觀塘綫', '荃灣綫', '港島綫', '將軍澳綫',
#               '東涌綫', '東鐵', '西鐵', '屯馬綫', '機場快綫', '巴士']
#    parking_keywords = ['parking', '停車', '泊車']
#    main_func(filename='hk_tweets_combined.csv',
#              interested_keywords=mtr_keywords,
#              accurate_position=False,
#              save_filename='MTR_related_tweets_filtered.csv')
#    main_func(filename='hk_tweets_combined.csv',
#              interested_keywords=parking_keywords,
#              accurate_position=False,
#              save_filename='parking_related_tweets_filtered.csv')
