# -*- coding: utf-8 -*-
# @Time    : 2022/11/17 11:43
# @Author  : Haoliang Chang
import os
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats

from data_paths import records_path


def paired_sample_t_test(first_array: np.ndarray, second_array: np.ndarray) -> Tuple[float, float]:
    """
    Performing the paired sample t-test.
    H0: It signifies that the mean first_array and second_array scores are equal
    HA: It signifies that the mean first_array and second_array scores are not equal
    :param first_array: The first numpy array
    :param second_array: The second numpy array
    :return: The paired sample t-test result
    """
    test_result = stats.ttest_rel(first_array, second_array)
    statistic_val, p_val = test_result.statistic, test_result.pvalue
    return statistic_val, p_val


def perform_t_test_overall_mobility(mobility_data_before: pd.DataFrame,
                                    mobility_data_during: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the paired sample t-test based on mobility metrics
    :param mobility_data_before:  a dataframe saving the mobility metrics before the pandemic
    :param mobility_data_during:  a dataframe saving the mobility metrics during the pandemic
    :return: a pandas dataframe saving the statistical test results
    """
    tweet_metrics_colnames = ['tweet_num', 'tweet_time_span']
    spatial_metrics_colname = ['mean_dist', 'std_dist', 'jump_dist', 'rg_val']
    temporal_metrics_colname = ['mean_time', 'std_time', 'jump_time']
    studied_colnames = tweet_metrics_colnames + spatial_metrics_colname + temporal_metrics_colname
    result_dict = dict()
    for colname in studied_colnames:
        before_vals, during_vals = np.array(mobility_data_before[colname]), np.array(mobility_data_during[colname])
        statistic_val, p_val = paired_sample_t_test(first_array=before_vals, second_array=during_vals)
        result_dict[colname] = [statistic_val, p_val]
    result_dataframe = pd.DataFrame(result_dict, index=['statistic_val', 'p_val'])
    return result_dataframe.round(4)


def test_poi_entropy():
    """
    Check whether the POI entropy changes significantly before and during the pandemic
    considered test: (one-sample or two-sample) Kolmogorov-Smirnov test
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    :return:
    """
    pass

def perform_t_test_individual_mobility():
    pass


if __name__ == '__main__':
    user_mobility_before_covid = pd.read_csv(os.path.join(records_path,
                                                          'user_mobility_before_covid.csv'),
                                             encoding='utf-8', index_col=0)
    user_mobility_during_covid = pd.read_csv(os.path.join(records_path, 'user_mobility_during_covid.csv'),
                                             encoding='utf-8', index_col=0)
    t_test_results = perform_t_test_overall_mobility(mobility_data_before=user_mobility_before_covid,
                                                     mobility_data_during=user_mobility_during_covid)
    t_test_results.to_csv(os.path.join(records_path, 't_test_mobility_results.csv'), encoding='utf-8')