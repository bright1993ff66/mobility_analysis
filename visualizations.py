"""
The visualizations.py saves some commonly used functions
for visualizations
"""
# basics
import os
from collections import Counter
import numpy as np
from typing import List
import random

# dataframe processing
import pandas as pd
from geopandas.geodataframe import GeoDataFrame

# for visualizations
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True  # enable formula in plots
from matplotlib import pyplot as plt

# other utilities
from data_paths import figures_path, records_path


def generate_colors(color_number: int) -> str or List:
    """
    Generate some random sampled colors
    :param color_number: the random color number you want to generate
    :return: a color string (if color_number = 1) or a list of colors
    (if color_number>1)
    """
    assert color_number >= 1, "You should at least generate one color"
    chars = '0123456789ABCDEF'
    if color_number == 1:
        colors = ['#' + ''.join(random.sample(chars, 6)) for _ in range(
            color_number)][0]
    else:
        colors = ['#' + ''.join(random.sample(chars, 6)) for _ in
                  range(color_number)]
    return colors


def setup_map_axis(ax: plt.axis, set_legend: bool = False) -> None:
    """
    Set up the axis of the map, including:
        - xtick and ytick sizes
        - xlabel and ylabel sizes
    :param ax: a subplot axis
    :param set_legend: whether you want to fix the legend
    :return: None. The axis is reformatted
    """
    # set the size of xticks and yticks
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    # set the size of x label and y label
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    # set the figure's axes to invisible
    # hide x-axis
    ax.get_xaxis().set_visible(False)
    # hide y-axis
    ax.get_yaxis().set_visible(False)
    # set the figure's spines to invisible
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if set_legend:  # Specify the size of text in legend
        ax.legend(fontsize=15)


def setup_figure_axis(ax: plt.axis, set_legend: bool = False) -> None:
    """
    Set up the axis in the subplots of matplotlib, including:
        - spines
        - xtick and ytick sizes
        - xlabel and ylabel sizes
    :param ax: a subplot axis
    :param set_legend: whether you want to fix the legend
    :return: None. The axis is reformatted
    """
    # set the figure's right and top spines to invisible
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set the size of xticks and yticks
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    # set the size of x label and y label
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    if set_legend:  # Specify the size of text in legend
        ax.legend(fontsize=15)


def get_ylim(axis: plt.axis) -> tuple:
    """
    Get the ylim of axis
    :param axis: one matplotlib axis
    :return: the ylim of one axis
    """
    return axis.get_ylim()


def get_xlim(axis: plt.axis) -> tuple:
    """
    Get the ylim of axis
    :param axis: one matplotlib axis
    :return: the ylim of one axis
    """
    return axis.get_xlim()


def define_box_properties(plot_name: dict, color_code: str, label: str, fontsize: int = None) -> None:
    """
    Define the properties for the box plot
    :param plot_name: the variable name of the boxplot created by axis.boxplot.
    This outcome is a dictionary
    :param color_code: the color code of the boxplot
    :param label: the label of the boxplot
    :param fontsize: the fontsize of legend
    :return: None.
    """
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)

    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    if fontsize:
        plt.legend(fontsize=fontsize, frameon=False)


def add_texts(x: list or np.ndarray, y: list or np.ndarray, ax: plt.axis) -> None:
    """
    Add the numbers to bar plots shown in one axis
    :param x: the values for the xlabel of the barchart
    :param y: the values for the ylabel of the barchart
    :param ax: a matplotlib axis (plt.axis)
    :return: None.
    """
    for i in range(len(x)):
        ax.text(x[i], y[i] + 0.01 * y[i], s=y[i], size=16)


def plot_inertias(inertia_dict, highest_cluster_num: int = 20) -> None:
    """
    Plot the inertia given an inertia dictionary
    :param inertia_dict: an inertia dictionary saving the cluster number and respective inertia values
    :param highest_cluster_num: the highest number of cluster number for consideration
    :return: A figure drawing the change of inertia values given cluster numbers
    """
    cluster_nums = list(inertia_dict.keys())
    considered_cluster_nums = list(filter(lambda val: val <= highest_cluster_num, cluster_nums))
    inertia_vals = [inertia_dict[cluster_num] for cluster_num in considered_cluster_nums]

    figure, axis = plt.subplots(1, 1, dpi=300)
    setup_figure_axis(axis)

    axis.plot(cluster_nums, inertia_vals, 'bx-')
    axis.set_xticks(list(range(3, highest_cluster_num+1, 1)))
    axis.set_xticklabels([str(val) for val in range(3, highest_cluster_num+1, 1)])
    axis.set_xlabel('Number of Clusters')
    axis.set_ylabel('Inertia')
    # axis.set_title('Select the Best Number of Clusters')
    figure.savefig(os.path.join(figures_path, 'select_user_cluster_num.png'),
                   bbox_inches='tight')


def plot_tweet_count_dist(count_dataframe: pd.DataFrame, percentile: float):
    """
    Plot the histogram of the number of tweets posted by users
    :param count_dataframe: a pandas dataframe saving the number of tweets posted by each user
    :param percentile: the interested percentile
    :return: None.
    """
    assert 'user_id' in count_dataframe, "The count dataframe saves the user id"
    assert 'count' in count_dataframe, "The count saves the number of appearance"
    assert 50 < percentile <= 99, "Please set an appropriate percentile: 50 < percentile <= 99"

    threshold = np.percentile(list(count_dataframe['count']), percentile)

    figure, axis = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
    count_dataframe['count'].hist(ax=axis, color='blue')
    axis.axvline(threshold, color='black')
    axis.text(200, 3000, "Threshold: {}".format(threshold))
    axis.grid(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def plot_geo_points_over_polygons(city_shape: GeoDataFrame,
                                  tweet_before_shape: GeoDataFrame,
                                  tweet_during_shape: GeoDataFrame,
                                  save_filename: str = None) -> None:
    """
    Plot the point features over the polygon features
    :param tweet_during_shape: the shapefile for the tweets posted before Covid19
    :param tweet_before_shape: the shapefile for the tweets posted during Covid19
    :param city_shape: the shapefile of the city boundary
    :param save_filename: the saved figure filename (default is None)
    :return: None. A figure depicting the points over the polygon feature is created
    """
    figure, axis = plt.subplots(dpi=300, figsize=(12, 10))
    # Specify the order of points and polygons using the "zorder" argument
    # The higher the zorder, the higher the level of features on the map
    city_shape.plot(ax=axis, color='white', edgecolor='black', zorder=1, linewidth=0.5)
    tweet_before_shape.plot(ax=axis, color='#1597A5', markersize=15, zorder=2, alpha=0.5,
                            label='Before Pandemic', marker='^')
    tweet_during_shape.plot(ax=axis, color='#FFC24B', markersize=15, zorder=3, alpha=0.5,
                            label='During Pandemic', marker='o')
    axis.set_xlabel('Longitude')
    axis.set_ylabel('Latitude')
    axis.legend(fontsize=20)
    setup_map_axis(axis)
    if save_filename:
        figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def plot_user_clusters(tweet_count_dataframe: pd.DataFrame,
                       cluster_labels: np.ndarray or list,
                       save_filename: str = None) -> None:
    """
    Plot the clusters of user ids based on time span and tweet count
    :param tweet_count_dataframe: a dataframe recording the number of tweets
    and tweet count time span
    :param cluster_labels: the cluster labels based on K-Means
    :param save_filename: the saved filename
    :return: None. A figure is created showing the cluster labels of
    Twitter users
    """
    # Double-check the column names
    count_columns = ['time_span', 'tweet_count']
    for colname in count_columns:
        if colname not in tweet_count_dataframe:
            raise ValueError('Column name {} not in dataframe'.format(colname))
    # assign the cluster label to the twitter id counter dataframe
    tweet_count_dataframe['cluster_label'] = cluster_labels
    tweet_count_arr = np.array(tweet_count_dataframe['tweet_count'])
    time_span_arr = np.array(tweet_count_dataframe['time_span'])
    # Visualize the user ids and cluster label
    cluster_figure, cluster_axis = plt.subplots(figsize=(10, 8), dpi=300)
    cluster_axis.scatter(x=time_span_arr, y=tweet_count_arr, s=10, c=cluster_labels,
                         cmap='Set3', edgecolors='None', alpha=1)
    cluster_axis.axvline(np.median(tweet_count_dataframe['time_span']), color='black',
                         alpha=0.5, linestyle='--', linewidth=0.5)
    cluster_axis.axhline(np.median(tweet_count_dataframe['tweet_count']), color='black',
                         alpha=0.5, linestyle='--', linewidth=0.5)
    setup_figure_axis(ax=cluster_axis)
    if not save_filename:
        cluster_figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def plot_footprint_mobility(mobility_metric_dataframe: pd.DataFrame,
                            color_code: str) -> None:
    """
    Plot the distance-based mobility metrics for the Twitter users. The dataframe can be generated by
    footprint_analysis.construct_user_mobility_data
    :param mobility_metric_dataframe: A pandas dataframe saving the mobility metrics of users,
    including mean and standard deviation of successive distances, whether jump exists, and
    radius of gyration.
    :param color_code: a text string specifies the color code of the histogram
    :return: None. The figure is saved to local directory
    """
    # Double-check the colnames
    interested_columns = ['mean_dist', 'std_dist', 'jump_dist', 'rg_val']
    for colname in interested_columns:
        assert colname in mobility_metric_dataframe, \
            "The dataframe should have a column named {}.".format(colname)
    # Count whether the jump exists
    jump_counter = Counter(mobility_metric_dataframe['jump_dist'])
    jump_vals = [0, 1]
    jump_exist_list = [jump_counter[0], jump_counter[1]]

    # Draw the plots
    footprint_figure, footprint_axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    footprint_axes[0, 0].hist(mobility_metric_dataframe['mean_dist'], color=color_code)
    footprint_axes[0, 1].hist(mobility_metric_dataframe['std_dist'], color=color_code)
    footprint_axes[1, 0].bar(x=jump_vals, height=jump_exist_list, color=color_code)
    footprint_axes[1, 1].hist(mobility_metric_dataframe['rg_val'], color=color_code)
    footprint_axes[0, 0].set_title('Mean Successive\nDistance', size=20)
    footprint_axes[0, 1].set_title('Standard Deviation of\nSuccessive Distance', size=20)
    footprint_axes[1, 0].set_title('Jump $\geq$ 10km Exists', size=20)
    footprint_axes[1, 1].set_title('Radius of Gyration', size=20)
    # Edit the xticks for the jump axis
    footprint_axes[1, 0].set_xticks([0, 1])
    footprint_axes[1, 0].set_xticklabels(["0", "1"])
    # Set the ylim for each axis

    # Format the axes
    setup_figure_axis(footprint_axes[0, 0])
    setup_figure_axis(footprint_axes[0, 1])
    setup_figure_axis(footprint_axes[1, 0])
    setup_figure_axis(footprint_axes[1, 1])
    footprint_figure.savefig(os.path.join(figures_path, 'user_mobility_metrics_distance.png'),
                             bbox_inches='tight')


def plot_temporal_mobility(mobility_metric_dataframe: pd.DataFrame,
                           color_code: str) -> None:
    """
    Plot the temporal mobility metrics for the Twitter users. The dataframe can be generated by
    footprint_analysis.construct_user_mobility_data
    :param mobility_metric_dataframe: A pandas dataframe saving the mobility metrics of users,
    including mean and standard deviation of successive distances, whether jump exists, and
    radius of gyration.
    :param color_code: a text string specifies the color code of the histogram
    :return: None. The figure is saved to local directory
    """
    # Double-check the colnames
    interested_columns = ['mean_time', 'std_time', 'jump_time']
    for colname in interested_columns:
        assert colname in mobility_metric_dataframe, \
            "The dataframe should have a column named {}.".format(colname)
    # Count whether the jump exists
    jump_counter = Counter(mobility_metric_dataframe['jump_time'])
    jump_vals = [0, 1]
    jump_exist_list = [jump_counter[0], jump_counter[1]]

    # Draw the plots
    temporal_figure, temporal_axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    temporal_axes[0, 0].hist(mobility_metric_dataframe['mean_time'], color=color_code)
    temporal_axes[0, 1].hist(mobility_metric_dataframe['std_time'], color=color_code)
    temporal_axes[1, 0].bar(x=jump_vals, height=jump_exist_list, color=color_code)
    temporal_axes[0, 0].set_title('Mean Successive\nTime Gaps', size=20)
    temporal_axes[0, 1].set_title('Standard Deviation of\nSuccessive Time Gaps', size=20)
    temporal_axes[1, 0].set_title('Jump $\geq$ 30 Days Exists', size=20)
    # set the axes[1,1] as invisible
    temporal_axes[1, 1].set_axis_off()  # remove the axis
    # Edit the xticks for the jump axis
    temporal_axes[1, 0].set_xticks([0, 1])
    temporal_axes[1, 0].set_xticklabels(["0", "1"])
    # Format the axes
    setup_figure_axis(temporal_axes[0, 0])
    setup_figure_axis(temporal_axes[0, 1])
    setup_figure_axis(temporal_axes[1, 0])
    temporal_figure.savefig(os.path.join(figures_path, 'user_mobility_metrics_time.png'),
                            bbox_inches='tight')


def plot_tweet_mobility(mobility_metric_dataframe: pd.DataFrame,
                        color_code: str) -> None:
    """
    Plot the mobility metrics for the Twitter users based on number of tweets and time span.
    The dataframe can be generated by footprint_analysis.construct_user_mobility_data
    :param mobility_metric_dataframe: A pandas dataframe saving the mobility metrics of users,
    including mean and standard deviation of successive distances, whether jump exists, and
    radius of gyration.
    :param color_code: a text string specifies the color code of the histogram
    :return: None. The figure is saved to local directory
    """
    # Double-check the colnames
    interested_columns = ['tweet_num', 'tweet_time_span']
    for colname in interested_columns:
        assert colname in mobility_metric_dataframe, \
            "The dataframe should have a column named {}.".format(colname)
    # Draw the plots
    tweet_figure, tweet_axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    tweet_axes[0].hist(mobility_metric_dataframe['tweet_num'], color=color_code)
    tweet_axes[1].hist(mobility_metric_dataframe['tweet_time_span'], color=color_code)
    tweet_axes[0].set_title('Number of Tweets', size=20)
    tweet_axes[1].set_title('Time Span of Posted Tweets', size=20)
    setup_figure_axis(tweet_axes[0])
    setup_figure_axis(tweet_axes[1])
    tweet_figure.savefig(os.path.join(figures_path, 'user_mobility_metrics_tweet.png'),
                         bbox_inches='tight')


def plot_footprint_mobility_cluster(mobility_metric_dataframe: pd.DataFrame,
                                    color_code: str = '#1597A5') -> None:
    """
    Plot the mobility metrics for the Twitter users of the same cluster. The dataframe can be
    generated by footprint_analysis.construct_user_mobility_data. The cluster label can be generated by
    activity_analysis.cluster_users
    :param mobility_metric_dataframe: A pandas dataframe saving the mobility metrics of users,
    including mean and standard deviation of successive distances, whether jump exists, and
    radius of gyration.
    :param color_code: a text string specifies the color code of the histogram
    :return: None. The figure is saved to local directory
    """
    # Count whether the jump exists
    jump_counter = Counter(mobility_metric_dataframe['jump_dist'])
    jump_vals = [0, 1]
    jump_exist_list = [jump_counter[0], jump_counter[1]]
    assert len(set(
        mobility_metric_dataframe['cluster'])) == 1, "Only one cluster should be considered!"
    cluster_label = list(mobility_metric_dataframe['cluster'])[0]

    # Draw the plots
    footprint_figure, footprint_axes = plt.subplots(2, 2, figsize=(10, 13), dpi=300)
    footprint_axes[0, 0].hist(mobility_metric_dataframe['mean_dist'], color=color_code)
    footprint_axes[0, 1].hist(mobility_metric_dataframe['std_dist'], color=color_code)
    footprint_axes[1, 0].bar(x=jump_vals, height=jump_exist_list, color=color_code)
    footprint_axes[1, 1].hist(mobility_metric_dataframe['rg_val'], color=color_code)
    footprint_axes[0, 0].set_title('Mean Successive\nDistance - Cluster {}'.format(
        cluster_label), size=20)
    footprint_axes[0, 1].set_title('Standard Deviation of\nSuccessive Distance - Cluster {}'.format(
        cluster_label), size=20)
    footprint_axes[1, 0].set_title('Jump $\geq$ 10km Exists - Cluster {}'.format(
        cluster_label), size=20)
    footprint_axes[1, 1].set_title('Radius of Gyration - Cluster {}'.format(cluster_label), size=20)
    setup_figure_axis(footprint_axes[0, 0])
    setup_figure_axis(footprint_axes[0, 1])
    setup_figure_axis(footprint_axes[1, 0])
    setup_figure_axis(footprint_axes[1, 1])
    footprint_figure.savefig(os.path.join(figures_path, 'user_cluster_{}.png'.format(
        cluster_label)), bbox_inches='tight')


def plot_metrics_for_one_cluster(mobility_before_covid: pd.DataFrame,
                                 mobility_during_covid: pd.DataFrame,
                                 cluster_num: int) -> None:
    """
    Plot the mobility metrics for one group of Twitter user before or during the pandemic
    :param mobility_before_covid: one dataframe containing the values of mobility metrics for
    each Twitter user before the pandemic
    :param mobility_during_covid: one dataframe containing the values of mobility metrics for
    each Twitter user during the pandemic
    :param cluster_num: the interested cluster num
    :return: None. The figure is saved to local directory
    """
    # Double-check the column of data
    assert 'cluster_before' in mobility_before_covid, "The data should contain cluster label"
    assert 'cluster_before' in mobility_during_covid, "The data should contain cluster label"
    data_colnames = ['mean_dist', 'std_dist', 'jump_dist', 'rg_val',
                     'mean_time', 'std_time', 'jump_time', 'tweet_num', 'tweet_time_span']
    for column in data_colnames:
        assert column in mobility_before_covid, "The data should contain a column named {}".format(column)
    for column in data_colnames:
        assert column in mobility_during_covid, "The data should contain a column named {}".format(column)
    # Set the saved filename and color
    save_filename_before = 'mobility_before_cluster_{}.png'.format(cluster_num)
    histogram_color_before = '#DF7A5E'
    save_filename_during = 'mobility_during_cluster_{}.png'.format(cluster_num)
    histogram_color_during = '#3C405B'
    print_message = 'Drawing mobility metrics for cluster {}'.format(cluster_num)
    rename_dict = {'mean_dist': 'Mean Successive\nDistance',
                   'std_dist': 'Standard Deviation of\nSuccessive Distance',
                   'jump_dist': 'Spatial Jump $\geq$ \n10km Exists',
                   'rg_val': 'Radius of Gyration',
                   'mean_time': 'Mean Successive\nTemporal Gaps',
                   'std_time': 'Standard Deviation of\nSuccessive Temporal Gaps',
                   'jump_time': 'Temporal Jump $\geq$ \n30 Days Exists',
                   'tweet_num': 'Number of Tweets',
                   'tweet_time_span': 'Tweet Time Span'}
    renamed_colnames = [rename_dict[colname] for colname in data_colnames]
    mobility_data_before_renamed = mobility_before_covid.rename(columns=rename_dict)
    mobility_data_during_renamed = mobility_during_covid.rename(columns=rename_dict)
    # Draw the figure
    print(print_message)
    before_figure, before_axis = plt.subplots(1, 1, dpi=300, figsize=(12, 12))
    during_figure, after_axis = plt.subplots(1, 1, dpi=300, figsize=(12, 12))
    mobility_cluster_before = mobility_data_before_renamed.loc[
        mobility_data_before_renamed['cluster_before'] == cluster_num]
    mobility_cluster_during = mobility_data_during_renamed.loc[
        mobility_data_during_renamed['cluster_before'] == cluster_num]
    num_users_before, num_tweets_before = mobility_cluster_before.shape[0], \
        sum(mobility_cluster_before['Number of Tweets'])
    num_users_during, num_tweets_during = mobility_cluster_during.shape[0], \
        sum(mobility_cluster_during['Number of Tweets'])
    assert num_users_before == num_users_during, "Number of users do not match!"
    print('Number of tweets before pandemic: {}; Number of Tweets during pandemic: {}'.format(
        num_tweets_before, num_tweets_during))
    before_axes = mobility_cluster_before[renamed_colnames].hist(
        ax=before_axis, color=histogram_color_before, grid=False)
    during_axes = mobility_cluster_during[renamed_colnames].hist(
        ax=after_axis, color=histogram_color_during, grid=False)
    # Align the ylim to the same range
    get_ylim_vals = np.vectorize(get_ylim)
    get_xlim_vals = np.vectorize(get_xlim)
    before_ylim_low, before_ylim_high = get_ylim_vals(before_axes)
    during_ylim_low, during_ylim_high = get_ylim_vals(during_axes)
    before_xlim_low, before_xlim_high = get_xlim_vals(before_axes)
    during_xlim_low, during_xlim_high = get_xlim_vals(during_axes)
    final_ylim_low = np.minimum(before_ylim_low, during_ylim_low)
    final_ylim_high = np.maximum(before_ylim_high, during_ylim_high)
    final_xlim_low = np.minimum(before_xlim_low, during_xlim_low)
    final_xlim_high = np.maximum(before_xlim_high, during_xlim_high)
    for (index_row, index_column), axis in np.ndenumerate(before_axes):
        axis.set_xlim(final_xlim_low[index_row, index_column], final_xlim_high[index_row, index_column])
        axis.set_ylim(final_ylim_low[index_row, index_column], final_ylim_high[index_row, index_column])
    for (index_row, index_column), axis in np.ndenumerate(during_axes):
        axis.set_xlim(final_xlim_low[index_row, index_column], final_xlim_high[index_row, index_column])
        axis.set_ylim(final_ylim_low[index_row, index_column], final_ylim_high[index_row, index_column])
    before_figure.suptitle('Before Covid19 - Number of Twitter Users in Cluster {}: {}\n Number of Tweets: {}'.format(
        cluster_num+1, num_users_before, num_tweets_before), size=25)
    during_figure.suptitle('During Covid19 - Number of Twitter Users in Cluster {}: {}\n Number of Tweets: {}'.format(
        cluster_num+1, num_users_during, num_tweets_during), size=25)
    # Save the figure to local directory
    before_figure.savefig(os.path.join(figures_path, save_filename_before), bbox_inches='tight')
    during_figure.savefig(os.path.join(figures_path, save_filename_during), bbox_inches='tight')


def create_plot_dist_time(mobility_data_before_covid: pd.DataFrame,
                          mobility_data_during_covid: pd.DataFrame,
                          plot_for_dist: bool = True) -> None:
    """
    Create the boxplot and distribution of jump exists variable based on distance metrics
    :param mobility_data_before_covid: the values of mobility metrics before Covid
    :param mobility_data_during_covid: the values of mobility metrics during Covid
    :param plot_for_dist: create a plot for distance metrics
    :return: None. The figure is saved to local directory
    """
    # Get the spatial and temporal mobility metrics
    mobility_before_continuous_dist = [list(mobility_data_before_covid['mean_dist']),
                                       list(mobility_data_before_covid['std_dist']),
                                       list(mobility_data_before_covid['rg_val'])]
    mobility_during_continuous_dist = [list(mobility_data_during_covid['mean_dist']),
                                       list(mobility_data_during_covid['std_dist']),
                                       list(mobility_data_during_covid['rg_val'])]
    mobility_before_continuous_time = [list(mobility_data_before_covid['mean_time']),
                                       list(mobility_data_before_covid['std_time'])]
    mobility_during_continuous_time = [list(mobility_data_during_covid['mean_time']),
                                       list(mobility_data_during_covid['std_time'])]
    jump_before_dist = list(mobility_data_before_covid['jump_dist'])
    jump_during_dist = list(mobility_data_during_covid['jump_dist'])
    jump_before_time = list(mobility_data_before_covid['jump_time'])
    jump_during_time = list(mobility_data_during_covid['jump_time'])

    # Create the figure object
    figure = plt.figure(figsize=(10, 15), dpi=300)
    boxplot_axis = plt.subplot(2, 1, 1)
    # Specify the properties of dots in the box plots
    flierprops = dict(marker='o', markerfacecolor='none', markersize=3,
                      markeredgecolor='black', alpha=0.5)

    if plot_for_dist:  # If plot for distance-based metrics
        print('Conduct data mining for distance-based metrics...')
        boxplot_ticks = ['Mean Successive\nDistance', 'Std Successive\nDistance',
                         'Radius of\nGyration']
        # Draw the box plots
        boxplot_before = boxplot_axis.boxplot(
            mobility_before_continuous_dist,
            positions=np.array(np.arange(len(mobility_before_continuous_dist))) * 2.0 - 0.35,
            widths=0.5, flierprops=flierprops)
        boxplot_during = boxplot_axis.boxplot(
            mobility_during_continuous_dist,
            positions=np.array(np.arange(len(mobility_during_continuous_dist))) * 2.0 + 0.35,
            widths=0.5, flierprops=flierprops)
        boxplot_axis.axhline(10, color='black', linestyle='--', alpha=0.5)
        xtick_median = np.percentile(boxplot_axis.xaxis.get_ticklocs(), 22)
        boxplot_axis.text(xtick_median, 10 + 10 * 0.12, 'Jump Threshold:\n 10km', size=15)
        jump_counter_before = Counter(jump_before_dist)
        jump_counter_during = Counter(jump_during_dist)
        boxplot_figure_ylabel = 'Kilometers'
        jump_figure_xlabel = 'Spatial Jump $\geq$ 10km Exists'
        jump_y_lim_val = 900  # setting the y lim for the jump exists variable
        legend_loc = 'upper right'
        save_filename = 'dist_mining.png'
    else:  # If plot for time-based metrics
        print('Conduct data mining for time-based metrics...')
        boxplot_ticks = ['Mean Successive\nTime Gaps', 'Std Successive\nTime Gaps']
        # Draw the box plots
        boxplot_before = boxplot_axis.boxplot(
            mobility_before_continuous_time,
            positions=np.array(np.arange(len(mobility_before_continuous_time))) * 2.0 - 0.35,
            widths=0.5, flierprops=flierprops)
        boxplot_during = boxplot_axis.boxplot(
            mobility_during_continuous_time,
            positions=np.array(np.arange(len(mobility_during_continuous_time))) * 2.0 + 0.35,
            widths=0.5, flierprops=flierprops)
        boxplot_axis.axhline(30, color='black', linestyle='--', alpha=0.5)
        xtick_median = np.percentile(boxplot_axis.xaxis.get_ticklocs(), 38)
        boxplot_axis.text(xtick_median, 30 + 30 * 0.1, 'Jump Threshold: 30 Days', size=15)
        jump_counter_before = Counter(jump_before_time)
        jump_counter_during = Counter(jump_during_time)
        boxplot_figure_ylabel = 'Num of Days'
        jump_figure_xlabel = 'Temporal Jump $\geq$ 30 days Exists'
        jump_y_lim_val = 1000  # setting the y lim for the jump exists variable
        legend_loc = 'upper left'
        save_filename = 'time_mining.png'

    # setting colors for each groups
    define_box_properties(boxplot_before, '#1597A5', 'Before Covid', fontsize=20)
    define_box_properties(boxplot_during, '#FFC24B', 'During Covid', fontsize=20)

    # Count whether the jump exists
    jump_vals = [0, 1]
    jump_exist_list_before = [jump_counter_before[0], jump_counter_before[1]]
    jump_exist_list_during = [jump_counter_during[0], jump_counter_during[1]]

    # Draw the figure showing the jump exists variable
    jump_axis_before_covid = plt.subplot(2, 2, 3)
    jump_axis_before_covid.bar(x=jump_vals,
                               height=jump_exist_list_before,
                               color='#1597A5', label='Before Covid')
    jump_axis_during_covid = plt.subplot(2, 2, 4)
    jump_axis_during_covid.bar(x=jump_vals,
                               height=jump_exist_list_during,
                               color='#FFC24B', label='During Covid')
    add_texts(jump_vals, jump_exist_list_before, ax=jump_axis_before_covid)
    add_texts(jump_vals, jump_exist_list_during, ax=jump_axis_during_covid)
    jump_axis_before_covid.legend(fontsize=12)
    jump_axis_during_covid.legend(fontsize=12)
    jump_axis_before_covid.set_ylim([0, jump_y_lim_val])
    jump_axis_during_covid.set_ylim([0, jump_y_lim_val])

    # Set the xtick values
    boxplot_axis.set_xticks(np.arange(0, len(boxplot_ticks) * 2, 2))
    boxplot_axis.set_xticklabels(boxplot_ticks)
    jump_axis_before_covid.set_xticks([0, 1])
    jump_axis_before_covid.set_xticklabels(["0", "1"])
    jump_axis_during_covid.set_xticks([0, 1])
    jump_axis_during_covid.set_xticklabels(["0", "1"])

    # Set the xlabel and ylabel
    boxplot_axis.set_ylabel(boxplot_figure_ylabel)
    jump_axis_before_covid.set_xlabel(jump_figure_xlabel)
    jump_axis_during_covid.set_xlabel(jump_figure_xlabel)
    jump_axis_before_covid.set_ylabel('Num of Twitter Users')
    jump_axis_during_covid.set_ylabel('Num of Twitter Users')

    # Format the axes
    setup_figure_axis(boxplot_axis, set_legend=True)
    setup_figure_axis(jump_axis_before_covid, set_legend=True)
    setup_figure_axis(jump_axis_during_covid, set_legend=True)
    boxplot_axis.legend(frameon=False)  # Remove the border in the legends
    jump_axis_before_covid.legend(frameon=False, loc=legend_loc)
    jump_axis_during_covid.legend(frameon=False, loc=legend_loc)

    # Save the figure to local directory
    figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def create_plot_tweet_metrics(mobility_data_before_covid: pd.DataFrame,
                              mobility_data_during_covid: pd.DataFrame) -> None:
    """
    Create the plot showing the tweet metrics.
    Reference:
    1. https://matplotlib.org/stable/gallery/statistics/boxplot_color.html
    2. https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color
    :param mobility_data_before_covid: the mobility dataframe before covid
    :param mobility_data_during_covid: the mobility dataframe during covid
    :return:
    """
    print('Conduct data mining for tweet-based metrics...')
    tweet_num_before_covid = [np.log(list(mobility_data_before_covid['tweet_num']))]  # use np.log to shrink data
    tweet_num_during_covid = [np.log(list(mobility_data_during_covid['tweet_num']))]  # use np.log to shrink data
    time_span_before_covid = [list(mobility_data_before_covid['tweet_time_span'])]
    time_span_during_covid = [list(mobility_data_during_covid['tweet_time_span'])]

    # Create the figure object
    figure = plt.figure(figsize=(20, 8), dpi=300)
    # Specify the properties of dots in the box plots
    flierprops = dict(marker='o', markerfacecolor='none', markersize=3,
                      markeredgecolor='black', alpha=0.5)

    # Draw the box plots
    tweet_num_axis = plt.subplot(1, 2, 1)
    boxplot_tweet_num_before = tweet_num_axis.boxplot(
        tweet_num_before_covid,
        positions=np.array(np.arange(len(tweet_num_before_covid))) * 2.0 - 0.35,
        widths=0.5, flierprops=flierprops)
    boxplot_tweet_num_during = tweet_num_axis.boxplot(
        tweet_num_during_covid,
        positions=np.array(np.arange(len(tweet_num_during_covid))) * 2.0 + 0.35,
        widths=0.5, flierprops=flierprops)
    time_span_axis = plt.subplot(1, 2, 2)
    boxplot_time_span_before = time_span_axis.boxplot(
        time_span_before_covid,
        positions=np.array(np.arange(len(time_span_before_covid))) * 2.0 - 0.35,
        widths=0.5, flierprops=flierprops)
    boxplot_time_span_during = time_span_axis.boxplot(
        time_span_during_covid,
        positions=np.array(np.arange(len(time_span_during_covid))) * 2.0 + 0.35,
        widths=0.5, flierprops=flierprops)

    # setting colors for each groups
    define_box_properties(boxplot_tweet_num_before, '#1597A5', 'Before Covid')
    define_box_properties(boxplot_tweet_num_during, '#FFC24B', 'During Covid')
    define_box_properties(boxplot_time_span_before, '#1597A5', 'Before Covid', fontsize=10)
    define_box_properties(boxplot_time_span_during, '#FFC24B', 'During Covid', fontsize=10)

    # set the xticks, x labels, and y labels
    tweet_num_axis.set_xticks([0])
    tweet_num_axis.set_xticklabels(["Tweet Number"])
    time_span_axis.set_xticks([0])
    time_span_axis.set_xticklabels(["Tweet Time Span"])
    tweet_num_axis.set_ylabel("log(Number of Tweets)")
    time_span_axis.set_ylabel('Number of Days')

    # Format the axes
    setup_figure_axis(tweet_num_axis, set_legend=False)
    setup_figure_axis(time_span_axis, set_legend=False)

    # Save the figure to local directory
    figure.savefig(os.path.join(figures_path, 'tweet_metrics.png'), bbox_inches='tight')


def main_visualization():
    """
    Main function to visualize the data
    :return: None. The figure is saved to local directory
    """
    mobility_before_covid = pd.read_csv(os.path.join(records_path,
                                                     'user_mobility_with_clusters_before.csv'),
                                        index_col=0, encoding='utf-8')
    mobility_during_covid = pd.read_csv(os.path.join(records_path,
                                                     'user_mobility_with_clusters_during.csv'),
                                        index_col=0, encoding='utf-8')
    # Plot the metrics for each cluster of user before and during the pandemic
    cluster_num = set(mobility_before_covid['cluster_before'])
    for cluster_id in cluster_num:
        plot_metrics_for_one_cluster(mobility_before_covid=mobility_before_covid,
                                     cluster_num=cluster_id,
                                     mobility_during_covid=mobility_during_covid)
    # Create the plot for overall spatial, temporal, tweet-based metrics
    create_plot_dist_time(mobility_data_before_covid=mobility_before_covid,
                          mobility_data_during_covid=mobility_during_covid,
                          plot_for_dist=True)
    create_plot_dist_time(mobility_data_before_covid=mobility_before_covid,
                          mobility_data_during_covid=mobility_during_covid,
                          plot_for_dist=False)
    create_plot_tweet_metrics(mobility_data_before_covid=mobility_before_covid,
                              mobility_data_during_covid=mobility_during_covid)


if __name__ == '__main__':
    main_visualization()
