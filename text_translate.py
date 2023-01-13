import os
from typing import Tuple
import numpy as np
import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from data_paths import tweet_combined_path
from utils import column_type_dict_filtered


# Load the tokenizer and model for translations
model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M")


def translate_text(text_string: str,source_lang: str) -> Tuple[str, str]:
    """
    Translate a text string to English text string
    """
    try:
        if source_lang != 'en' and source_lang != 'und' and source_lang != 'in':
            tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            tokenizer.src_lang = source_lang
            encoded_text = tokenizer(text_string, return_tensors="pt")
            generated_tokens = model.generate(**encoded_text,
                                              forced_bos_token_id=tokenizer.get_lang_id("en"))
            translation = tokenizer.batch_decode(generated_tokens,
                                                 skip_special_tokens=False)[0][7:]
            print('Original text: {}'.format(text_string))
            print('Translation: {}'.format(translation))
            return translation, 'translation nothing wrong'
        elif source_lang == 'und':
            return text_string, 'lang undefined'
        elif source_lang == 'in':
            tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            encoded_text = tokenizer(text_string, return_tensors="pt")
            generated_tokens = model.generate(**encoded_text,
                                              forced_bos_token_id=tokenizer.get_lang_id("en"))
            translation = tokenizer.batch_decode(generated_tokens,
                                                 skip_special_tokens=False)[0][7:]
            print('Original text: {}'.format(text_string))
            print('Translation: {}'.format(translation))
            return translation, 'language code in'
        else:
            return text_string, 'English text'
    except KeyError:
        print('KeyError happens for the language type: {}'.format(
            source_lang))
        return text_string, 'lang cannot tranlsated'


def main_translation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Main function for translation
    """
    dataframe_reindex = dataframe.reset_index(drop = True).copy()
    translated_list, message_list = [], []
    for index, row in dataframe_reindex.iterrows():
        print('Coping with the {}th row, {}'.format(
            index, row['lang']))
        translation, message = translate_text(str(row['text']), row['lang'])
        translated_list.append(translation)
        message_list.append(message)
    dataframe_reindex['translate'] = translated_list
    dataframe_reindex['message'] = message_list
    return dataframe_reindex


if __name__ == '__main__':
    print('Load the dataframes...')
#    hk_2018_tweets_common = pd.read_csv(os.path.join(
#        tweet_combined_path, 'hk_tweets_2018_from_common_users.csv'),
#                                        index_col=0)
    hk_covid_tweets_common = pd.read_csv(os.path.join(
        tweet_combined_path, 'hk_tweets_current_from_common_users.csv'),
                                        index_col=0,
        dtype=column_type_dict_filtered)
    print('Done! Start translation...')
#    hk_2018_tweets_common_translated = main_translation(hk_2018_tweets_common)
#    hk_2018_tweets_common_translated.to_csv(
#        os.path.join(tweet_combined_path, 'hk_common_2018_translated.csv'),
#        encoding='utf-8')
    hk_covid_tweets_common_translated = main_translation(hk_covid_tweets_common)
    hk_covid_tweets_common_translated.to_csv(
        os.path.join(tweet_combined_path, 'hk_common_covid_translated.csv'),
        encoding='utf-8')
