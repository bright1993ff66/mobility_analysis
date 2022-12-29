# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 20:07
# @Author  : Haoliang Chang
import os
import time
import numpy as np
import pandas as pd

from translate import Translator
from data_paths import tweet_combined_path


def translate_text(text_string: str, source_lang: str) -> str:
    """
    Translate a text string to English using the translate package
    ref: https://github.com/terryyin/translate-python
    :param text_string: a text string
    :param source_lang: the source language
    :return: the translated language
    """
    if source_lang != 'en':
        translator = Translator(to_lang="en", from_lang=source_lang)
        translation = translator.translate(text_string)
    else:
        translation = text_string
    return translation


def translate_tweet_dataframe(tweet_dataframe: pd.DataFrame,
                              sleep_threshold=500) -> pd.DataFrame:
    """
    Translate the tweet dataframe
    :param tweet_dataframe: a tweet dataframe
    :param sleep_threshold: the threshold we set to sleep
    :return: the tweet dataframe with translated text
    """
    translated_dataframe = tweet_dataframe.copy().reset_index(drop=True)
    print('In total, we have {} rows.'.format(translated_dataframe.shape[0]))
    # translated_dataframe['trans'] = translated_dataframe.apply(
    #     lambda row: translate_text(text_string=row['text'], source_lang=row['lang']), axis=1)
    original_text, translations = [], []
    for index, row in translated_dataframe.iterrows():
        print('Coping with the {}th row'.format(index + 1))
        text_to_be_translated = str(row['text'])
        original_text.append(text_to_be_translated)
        source_language = row['lang']
        print('Translating text: {}; source language: {}'.format(
            text_to_be_translated[:10], source_language))
        translation = translate_text(text_string=text_to_be_translated,
                                      source_lang=source_language)
        translations.append(translation)
        if index and (not index % sleep_threshold):
            random_sec = np.random.randint(low=20, high=50)
            print('Sleep for: {} secs'.format(random_sec))
            time.sleep(random_sec)
            print('Done! Start translation...')
    translated_dataframe['trans'] = translations
    return translated_dataframe


if __name__ == '__main__':
    hk_tweets_2018_common = pd.read_csv(os.path.join(tweet_combined_path,
                                                     'combined_hk_tweets/hk_tweets_2018_from_common_users.csv'),
                                        index_col=0)
    translated_tweets_2018 = translate_tweet_dataframe(tweet_dataframe=hk_tweets_2018_common)
    translated_tweets_2018.to_csv(os.path.join(tweet_combined_path,
                                               'hk_tweets_2018_common_translated.csv'),
                                  encoding='utf-8')
