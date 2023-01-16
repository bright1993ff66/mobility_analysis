"""
The utils.py saves some commonly used functions to process tweets
"""
# basics
import os
from collections import Counter
from typing import Tuple
from random import sample

# Cope with dates
import pytz
from datetime import datetime

# dataframe processing
import numpy as np
import pandas as pd
from pandas import ExcelWriter

# for spatial analysis
import geopandas as gpd
from shapely.geometry import Point

# Get the timezone of Hong Kong
timezone_hongkong = pytz.timezone('Asia/Hong_Kong')

# Specify the random seed
random_seed = 777

# The project coordinate system of Hong Kong
hong_kong_epsg = 2326

# Specify the interested columns and their data types when reading raw tweets
column_dtype_dict = {'user_id_str': str, 'id_str': str, 'text': str,
                     'created_at': str, 'lat': str, 'lon': str,
                     'place_lat': np.float64, 'place_lon': np.float64,
                     'verified': str, 'lang': str, 'url': str,
                     'country_code': str}
considered_columns = list(column_dtype_dict.keys())

# types of columns for the filtered tweets
column_type_dict_filtered = {'text': str, 'id_str': str, 'created_str': str, 'lang': str,
                             'verified': str, 'location': str, 'user_id_str': str, 'country_code': str,
                             'place_lat': np.float64, 'place_lon': np.float64, 'lat': np.float64,
                             'lon': np.float64, 'url': str, 'hk_time': str, 'year': int, 'month': int,
                             'day': int, 'hour': int, 'minute': int, 'geometry': str, 'index_right': str,
                             'FID_1': str, 'merge_Nums': str, 'SmallTPU': str, 'message': str,
                             'translate': str}


def return_max_key(dictionary: dict):
    """
    Get the key for a dictionary's max value
    :param dictionary: a python dictionary with float values
    :return: the key with maximum value
    """
    return max(dictionary, key=dictionary.get)


def get_epsg_code(shape) -> int:
    """
    Get the epsg code given a shapefile
    :param shape: a shapefile
    :return: the epsg code
    """
    return shape.crs.to_epsg()


def transform_shapefiles(target_epsg: int) -> None:
    """
    Transform the coordinate systems of all shapefiles in current
    directory to a target crs
    :param target_epsg: the epsg code of target crs
    :return: None. The shapefiles are saved to local directory
    """
    shapefile_names = [file for file in os.listdir(
        os.getcwd()) if file.endswith('.shp')]
    for file_name in shapefile_names:
        print('Transforming the file: {} to epsg {}'.format(file_name, target_epsg))
        shapefile = gpd.read_file(file_name, encoding='utf-8')
        shapefile_transform = shapefile.to_crs(epsg=target_epsg)
        shapefile_transform.to_file(file_name, encoding='utf-8')


def transform_string_time_to_datetime(time_string, target_time_zone,
                                      convert_utc_time=False):
    """
    Transform the string time to the datetime
    :param time_string: a time string
    :param target_time_zone: the target time zone
    :param convert_utc_time: whether transfer the datetime object to utc first.
    This is true when the time string is recorded as the UTC time
    :return: a structured datetime object
    """
    datetime_object = datetime.strptime(time_string, '%a %b %d %H:%M:%S %z %Y')
    if convert_utc_time:
        final_time_object = datetime_object.replace(
            tzinfo=pytz.utc).astimezone(target_time_zone)
    else:
        final_time_object = datetime_object.astimezone(target_time_zone)
    return final_time_object


def transform_datetime_string_to_datetime(
        time_string: str, target_timezone: pytz.tzfile = timezone_hongkong) -> datetime:
    """
    Transform a datetime string to the corresponding datetime. The source
    timezone is in +8:00
    **WARNING**: Don't use this function if the summer time is considered
    :param time_string: the string which records the time of the posted tweets
    (this string's timezone is HK time)
    :param target_timezone: the target timezone datetime object
    :return: a datetime object which could get access to the year, month,
    day easily
    """
    datetime_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S+08:00')
    final_time_object = target_timezone.localize(datetime_object)
    return final_time_object


def compute_time_span(tweet_dataframe: pd.DataFrame) -> int:
    """
    Compute the time span between the first tweet and last tweet
    :param tweet_dataframe: a tweet dataframe recording all the tweets posted by one user
    :return: the time span between the first tweet and the last tweet
    """
    assert 'hk_time' in tweet_dataframe, "The point data should have a column named 'hk_time'"
    if type(list(tweet_dataframe['hk_time'])[0]) == str:
        tweet_dataframe['hk_time'] = tweet_dataframe.apply(
            lambda row: transform_datetime_string_to_datetime(
                row['hk_time'], target_timezone=timezone_hongkong), axis=1)
    time_list = list(tweet_dataframe['hk_time'])
    first_time, last_time = time_list[0], time_list[-1]
    return (last_time - first_time).days


def filter_rows_having_strings(dataframe: pd.DataFrame,
                               string_list: list,
                               text_colname: str) -> pd.DataFrame:
    """
    Filter the dataframe based on a list of strings
    param: dataframe a pandas dataframe
    param: string_list a list containing the interested strings
    param: text_colname: the colname of text string
    return: a pandas filtered dataframe
    """
    dataframe_filtered = dataframe.loc[
        dataframe[text_colname].str.contains('|'.join(string_list))]
    return dataframe_filtered


def combine_some_data(path: str, sample_num: int = None,
                      get_geocoded: bool = True,
                      considered_users = None) -> pd.DataFrame:
    """
    Combine some random sampled dataframes from a local path
    :param path: an interested path
    :param sample_num: the number of files we want to consider
    :param get_geocoded: only get the geocoded tweets or not
    :param considered_users: a set containing the ids of considered users
    :return a pandas dataframe saving the tweets
    """
    if considered_users is None:
        considered_users = set()
    files = [file for file in os.listdir(path) if file.endswith('.csv')]
    if not sample_num:
        random_sampled_files = files
    else:
        random_sampled_files = sample(files, k=sample_num)

    dataframe_list = []
    for file in random_sampled_files:
        print('Coping with the file: {}'.format(file))
        # No need to add index_col=0 if considered columns are given
        try:
            dataframe = pd.read_csv(os.path.join(
                path, file), encoding='utf-8',
                usecols=considered_columns,
                dtype=column_dtype_dict)
        except UnicodeDecodeError:
            dataframe = pd.read_csv(open(os.path.join(path, file),
                                         errors='ignore', encoding='utf-8'),
                                    usecols=considered_columns,
                                    dtype=column_dtype_dict)
        # Set the data type of columns
        dataframe['user_id_str'] = dataframe['user_id_str'].astype(float)
        dataframe['lat'] = dataframe['lat'].astype(float)
        dataframe['lon'] = dataframe['lon'].astype(float)
        dataframe['place_lat'] = dataframe['place_lat'].astype(float)
        dataframe['place_lon'] = dataframe['place_lon'].astype(float)
        # Only consider subset of user
        if considered_users:
            dataframe_from_users = dataframe.loc[
                dataframe['user_id_str'].isin(considered_users)]
        else:
            dataframe_from_users = dataframe.copy()
        if get_geocoded:  # Only consider the geocoded tweets
            dataframe_geocoded = dataframe_from_users[~dataframe_from_users['lat'].isna()]
            dataframe_list.append(dataframe_geocoded)
        else:  # Consider all the tweets
            dataframe_geocoded = dataframe_from_users[~dataframe_from_users['place_lat'].isna()]
            dataframe_list.append(dataframe_geocoded)

    concat_dataframe = pd.concat(dataframe_list, axis=0)
    concat_dataframe_reindex = concat_dataframe.reset_index(drop=True)
    return concat_dataframe_reindex


def write_to_excel(dataframe: pd.DataFrame, save_filename: str,
                   save_path: str = None) -> None:
    """
    Write the dataframe to an Excel file
    :param dataframe: a tweet pandas dataframe
    :param save_filename: the filename of saved Excel file
    :param save_path: the path to the saved file
    :return None. The created Excel file is saved to a local directory
    """
    assert 'xlsx' in save_filename, "The filename should end with .xlsx"
    if not save_path:
        with ExcelWriter(save_filename) as writer:
            dataframe.to_excel(writer)
    else:
        with ExcelWriter(os.path.join(save_path, save_filename)) as writer:
            dataframe.to_excel(writer)


def number_of_tweet_users(dataframe: pd.DataFrame,
                          print_value: bool = False) -> None or Tuple:
    """
    Get the number of tweets and number of Twitter users
    :param dataframe: the studied dataframe
    :param print_value: whether print the values or not
    :return: the number of tweets and number of users
    """
    assert 'user_id_str' in dataframe, "Miss the user id info"
    assert 'id_str' in dataframe, "Miss the tweet id info"

    number_of_tweets = len(set(dataframe['id_str']))
    number_of_users = len(set(dataframe['user_id_str']))
    if print_value:
        print('The number of tweets: {} The number of unique social media users: {}'.format(
            number_of_tweets, number_of_users))
    else:
        return number_of_tweets, number_of_users


def spatial_join(point_gdf: gpd.geodataframe, shape_area: gpd.geodataframe) -> gpd.geodataframe:
    """
    Find the tweets posted in one city's considered boundary (e.g., city boundary and open space)
    :param point_gdf: the geopandas dataframe saving the tweets or crashes.
    :param shape_area: the shapefile of a studied area, such as city, open space, etc.
    :return: points posted in the considered area.
    """
    if not point_gdf.crs:  # If the point feature does not have coordinate system
        shape_epsg_code = get_epsg_code(shape_area)  # Use the epsg code from the map
        point_gdf = point_gdf.set_crs(epsg=shape_epsg_code)
    assert point_gdf.crs == shape_area.crs, 'The coordinate systems do not match!'

    joined_data = gpd.sjoin(left_df=point_gdf, right_df=shape_area, predicate='within')
    if 'id_str' in joined_data:
        joined_data_final = joined_data.drop_duplicates(subset=['id_str'])
    else:
        joined_data_final = joined_data.copy()
    return joined_data_final


def find_nearby_pois(geo_tweet_lat: float, geo_tweet_lon: float,
                     poi_data: gpd.geodataframe.GeoDataFrame,
                     set_point_4326: bool = True,
                     point_epsg: int = hong_kong_epsg,
                     buffer_radius: float = 500):
    """
    Find the nearby POIs based on tweet location. 'x' means longitude and 'y'
    represents latitude
    :param geo_tweet_lat: the latitude of the geocoded tweet
    :param geo_tweet_lon: the longitude of the geocoded tweet
    :param poi_data: the Point-of-Interest (POI) data
    :param set_point_4326: whether set the coordinate system of point feature to
    epsg=4326 first before processing
    :param point_epsg: the epsg code of point
    :param buffer_radius: the radius of buffer to find the nearby POIs. Default: 500 meter
    :return: the POIs close to the geotagged tweets
    """
    # Generate the shapely point (put the longitude before the latitude in the Point geometry)
    if set_point_4326:  # If the lat and lon of points are from epsg=4326
        tweet_point = gpd.GeoSeries(Point(geo_tweet_lon, geo_tweet_lat)).set_crs(
            epsg=4326).to_crs(epsg=point_epsg)
    else:  # Otherwise
        tweet_point = gpd.GeoSeries(Point(geo_tweet_lon, geo_tweet_lat)).set_crs(
            epsg=point_epsg)
    # Generate the buffer based on the geocoded tweet - 500 meter
    tweet_buffer = gpd.GeoDataFrame(gpd.GeoSeries(tweet_point.buffer(buffer_radius)),
                                    columns=['geometry'])
    # Find the POIs within the buffer
    selected_pois = spatial_join(point_gdf=poi_data, shape_area=tweet_buffer)
    return selected_pois


def create_geodataframe_from_csv(dataframe: pd.DataFrame, source_crs: int,
                                 target_crs: int,
                                 accurate_pos: bool = True) -> gpd.geodataframe.GeoDataFrame:
    """
    Create the geopandas geodataframe from a pandas dataframe
    :param dataframe: a pandas dataframe
    :param source_crs: the source coordinate system (epsg code)
    :param target_crs: the target coordinate system (epsg code)
    :param accurate_pos: whether you use lat & lon instead of place_lat
    & place_lon
    :return: a geopandas dataframe that can be used for spatial analysis
    """
    assert 'lat' in dataframe or 'place_lat' in dataframe, \
        "The dataframe should contain the geo-information"
    if accurate_pos:
        dataframe_geo = gpd.GeoDataFrame(
            dataframe, geometry=gpd.points_from_xy(dataframe.lon,
                                                   dataframe.lat))
    else:
        dataframe_geo = gpd.GeoDataFrame(
            dataframe, geometry=gpd.points_from_xy(dataframe.place_lon,
                                                   dataframe.place_lat))
    dataframe_geo = dataframe_geo.set_crs(epsg=source_crs, inplace=True)
    if source_crs != target_crs:
        dataframe_geo_final = dataframe_geo.to_crs(epsg=target_crs)
    else:
        dataframe_geo_final = dataframe_geo.copy()
    return dataframe_geo_final


def get_time_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the year, month, and day information
    :param df: a pandas dataframe saving the tweets
    :return: a tweet dataframe with year, month, and day information
    """
    assert 'hk_time' in df, "The dataframe should have a column named hk_time"
    df_copy = df.copy()
    df_copy['year'] = df_copy.apply(
        lambda row: row['hk_time'].year, axis=1)
    df_copy['month'] = df_copy.apply(
        lambda row: row['hk_time'].month, axis=1)
    df_copy['day'] = df_copy.apply(
        lambda row: row['hk_time'].day, axis=1)
    df_copy['hour'] = df_copy.apply(
        lambda row: row['hk_time'].hour, axis=1)
    df_copy['minute'] = df_copy.apply(
        lambda row: row['hk_time'].minute, axis=1)
    return df_copy


def prepare_tweet_data(tweet_dataframe_csv: pd.DataFrame,
                       choose_accurate_pos: bool = False) -> pd.DataFrame:
    """
    Prepare the spatial & temporal information of tweets for following analysis
    :param tweet_dataframe_csv: a tweet dataframe loaded from csv
    :param choose_accurate_pos: whether we use accurate position or not
    :return: a pandas dataframe with transformed hk_time and location information
    """
    # Process the temporal information
    assert 'hk_time' in tweet_dataframe_csv, "The point data should have a column named 'hk_time'"
    if isinstance(list(tweet_dataframe_csv['hk_time'])[0], str):
        tweet_dataframe_csv['hk_time'] = tweet_dataframe_csv.apply(
            lambda row: transform_datetime_string_to_datetime(row['hk_time']), axis=1)
    # Process the location information
    tweet_final = create_geodataframe_from_csv(dataframe=tweet_dataframe_csv, source_crs=4326,
                                               target_crs=hong_kong_epsg,
                                               accurate_pos=choose_accurate_pos)
    return tweet_final


def general_info_of_tweet_dataset(df: pd.DataFrame,
                                  study_area: str = 'Hong_Kong') -> None:
    """
    Get the general info of a tweet dataframe
    :param df: a pandas tweet dataframe
    :param study_area: a string describing the study area
    :return: None. A short description of the tweet dataframe is given,
    including user number, tweet number,
    average number of tweets per day, language distribution and sentiment
    distribution
    """
    assert 'hk_time' in df, "A column named hk_time is missing"
    assert 'user_id_str' in df, "Miss the user id info"
    assert 'id_str' in df, "Miss the tweet id info"

    df_copy = df.copy()
    if type(list(df_copy['hk_time'])[0]) == str:
        df_copy['hk_time'] = df_copy.apply(
            lambda row: transform_datetime_string_to_datetime(
                row['hk_time'], target_timezone=timezone_hongkong), axis=1)
    df_sorted = df_copy.sort_values(by='hk_time')
    user_number = len(set(df_sorted['user_id_str']))
    tweet_number = len(set(df_sorted['id_str']))
    starting_time = list(df_sorted['hk_time'])[0]
    ending_time = list(df_sorted['hk_time'])[-1]
    daily_tweet_count = df_sorted.shape[0] / (ending_time - starting_time).days
    language_dist_dict = Counter(df_sorted['lang'])

    print('For {}\n'.format(study_area))
    print('Number of users: {}\n'.format(user_number))
    print('Number of tweets: {}\n'.format(tweet_number))
    print('Avg daily num of tweets: {}\n'.format(daily_tweet_count))
    print("Language distribution: {}".format(language_dist_dict))


if __name__ == "__main__":
    print('The considered columns are: {}'.format(considered_columns))
