"""
This modules saves the paths for this project
"""
import os

Project_path = r'D:\Projects\hk_covid'
tweet_path = os.path.join(Project_path, 'hk_tweets')
tweet_prev = os.path.join(Project_path,
                          'hk_tweets_prev',
                          'HongKong', '2018')
figures_path = os.path.join(Project_path, 'figures')
records_path = os.path.join(Project_path, 'data')
tweet_combined_path = os.path.join(Project_path, 'combined_hk_tweets')
shapefile_path = os.path.join(Project_path, 'shapefiles')
topic_results = os.path.join(Project_path, 'topic_model')


if __name__ == '__main__':
    print("The home directory is: {}".format(Project_path))
