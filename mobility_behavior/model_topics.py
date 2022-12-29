# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 20:30
# @Author  : Haoliang Chang
import os
from random import sample
from string import punctuation
import numpy as np
import pandas as pd

# For text processing and topic modeling
import re
import gensim
# import spacy
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from wordcloud import STOPWORDS

# Load the shapefiles
import geopandas as gpd

# Variables and functions from other modules
from data_paths import topic_results, tweet_combined_path
from data_paths import shapefile_path, records_path
from utils import random_seed
from utils import find_nearby_pois

# # Load the tokenizer in SpaCy
# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Add the unuseful terms
stopwords = list(set(STOPWORDS))
strange_terms = ['allcaps', 'repeated', 'elongated', 'repeat', 'user', 'percent_c', 'hong kong', 'hong',
                 'kong', 'u_u', 'u_u_number', 'u_u_u_u', 'u_number', 'elongate', 'u_number_u',
                 'u', 'number', 'm', 'will', 'hp', 'grad', 'ed', 'boo']
unuseful_terms = stopwords + strange_terms
unuseful_terms_set = set(unuseful_terms)

# Specify the search parameters of lda
# Set the hyperparameter: the number of the topics
topic_modelling_search_params = {'n_components': [5, 6, 7, 8, 9, 10]}


def remove_non_alphabetical_letters(text_string: str) -> str:
    """
    Remove the non-alphabetical letters from a text string
    :param text_string: one text string
    :return: a cleaned text string
    """
    # Remove the url
    text_without_url = re.sub(r'https?:\/\/.*[\r\n]*', '', text_string)
    # Remove the non-alphabetical letters
    new_text_list = []
    text_list = text_without_url.split(' ')
    for text in text_list:
        cleaned_text = re.sub("[^a-zA-Z0-9 {}]".format(punctuation), '', text)
        new_text_list.append(cleaned_text)
    final_text_list = [word for word in new_text_list if len(word) > 0]
    return ' '.join(final_text_list)


def process_words(texts: str, stop_words: set,
                  bigram_mod: gensim.models.phrases.Phraser,
                  trigram_mod: gensim.models.phrases.Phraser) -> list:
    """
    Remove Stopwords, Form Bi-grams, Trigrams and Lemmatization
    """
    texts = [remove_non_alphabetical_letters(text_string=text) for text in texts]
    texts_list = [tweet.split(' ') for tweet in texts]
    texts_list = [[word for word in doc if word not in stop_words] for doc in texts_list]
    texts_list = [bigram_mod[doc] for doc in texts_list]
    texts_list = [trigram_mod[bigram_mod[doc]] for doc in texts_list]
    # texts_out = []
    # for sent in texts:
    #     doc = nlp(" ".join(sent))
    #     texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    sampled_texts = sample(texts_list, 2)
    print('Some sampled processed text: {}, \n{}'.format(sampled_texts[0], sampled_texts[1]))
    return texts_list


def process_poi_names(poi_dataframe: gpd.geodataframe.GeoDataFrame) -> list:
    """
    Get the class names of POIs
    :param poi_dataframe: a geopandas dataframe saving the categories of each POI
    :return: a list contained the POI classes
    """
    assert 'fclass' in poi_dataframe, "The POI dataframe should contain a column named 'fclass'"
    poi_class_list = list(poi_dataframe['fclass'])
    return poi_class_list


# Show top n keywords for each topic
def show_topics(vectorizer: CountVectorizer, lda_model: LatentDirichletAllocation, n_words: int = 20) -> list:
    """
    Show the keywords of topics generated by LDA model
    :param vectorizer: a CountVectorizer from sklearn.feature_extraction.text.CountVectorizer
    :param lda_model: a LatentDirichletAllocation model
    :param n_words: the number of keywords considered for each topic
    :return: a list containing the keywords of topics
    """
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


def get_lda_model(text_in_one_list, grid_search_params, number_of_keywords, topic_predict_file,
                  keywords_file, topic_number, grid_search_or_not=True,
                  saving_path=topic_results):
    """
    :param text_in_one_list: a text list. Each item of this list is a posted tweet
    :param grid_search_params: the dictionary which contains the values of hyperparameters for grid search
    :param number_of_keywords: number of keywords to represent a topic
    :param topic_predict_file: one file which contains the predicted topic for each tweet
    :param keywords_file: one file which saves all the topics and keywords
    :param topic_number: The number of topics we use(this argument only works if grid_search_or_not = False)
    :param grid_search_or_not: Whether grid search to get 'best' number of topics
    :param saving_path: path used to save the results
    """
    # 1. Vectorized the data
    # For more info, please check:
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=1,  # minimum occurrences of a word
                                 stop_words='english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=50000,   # max number of unique words
                                 )
    text_vectorized = vectorizer.fit_transform(text_in_one_list)

    # 2. Use the GridSearch to find the best hyperparameter
    # In this case, the number of topics is the hyperparameter we should tune
    lda = LatentDirichletAllocation(learning_method='batch', random_state=random_seed)
    if not grid_search_or_not and topic_number:
        model = GridSearchCV(lda, param_grid={'n_components': [topic_number]})
    else:
        model = GridSearchCV(lda, param_grid=grid_search_params)
    model.fit(text_vectorized)
    # See the best model
    best_lda_model = model.best_estimator_
    if grid_search_or_not:
        # Model Parameters
        print("Best Model's Params: ", model.best_params_)
        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)
        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(text_vectorized))
    else:
        # Show the number of topics we use
        print('The number of topics we use: {}'.format(topic_number))
        # Log likelihood score
        print('The log-likelihood score is {}'.format(model.best_score_))
        # Perplexity
        print("Model Perplexity: {}".format(best_lda_model.perplexity(text_vectorized)))

    # 3. Use the best model to fit the data
    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(text_vectorized)
    # column names
    topic_names = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
    # index names
    doc_names = ["Tweet" + str(i) for i in range(np.shape(text_vectorized)[0])]
    # Make the pandas dataframe
    # The df_document_topic dataframe just shows the dominant topic of each doc(tweet)
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topic_names, index=doc_names)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    df_document_topic.to_csv(os.path.join(saving_path, topic_predict_file), encoding='utf-8')
    # Apply Style
    # df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    # df_document_topics
    # Show the number of topics appeared among documents
    # df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    # df_topic_distribution.columns = ['Topic Num', 'Num Documents']

    # 4. Get the keywords for each topic
    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=number_of_keywords)
    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    df_topic_keywords.to_csv(os.path.join(saving_path, keywords_file), encoding='utf-8')


def develop_lda_pois(coordinates_records: gpd.geodataframe.GeoDataFrame,
                     poi_dataframe: gpd.geodataframe.GeoDataFrame,
                     keyword_file_name: str, topic_number: int, topic_predict_file_name: str,
                     saving_path=topic_results,
                     search_radius: float = 500.0,
                     keyword_num: int = 15):
    """
    Understand the surroundings of each geotagged tweets based on POI data
    :param coordinates_records: a geodataframe or dataframe saving the coordinates shared by Twitter users
    :param poi_dataframe: a POI dataframe
    :param keyword_file_name: the name of the saved file which contains the keyword for each topic
    :param topic_number: the number of topics we set for the topic modelling
    :param topic_predict_file_name: the name of the saved file which contains the topic prediction for
    each tweet
    :param saving_path: the saving path
    :param search_radius: a search radius to find the nearby Point of interests
    :param keyword_num: the number of keywords used to express one topic
    :return: LDA topic modeling results based on surrounding POIs
    """
    # Double-check the settings of tweet coordinates and POI dataframe
    assert 'lat' in coordinates_records or 'place_lat' in coordinates_records, \
        "The dataframe should contain the geo-information"
    assert coordinates_records.crs == poi_dataframe.crs, \
        "The coordinate systems do not match!"
    assert 'fclass' in poi_dataframe, "The poi data should have class information"
    lat_colname, lon_colname = 'place_lat', 'place_lon'
    # Find the class of POIs near each geocoded tweets
    poi_text_list = []
    for index, row in coordinates_records.iterrows():
        tweet_lat, tweet_lon = row[lat_colname], row[lon_colname]
        print('Finding POIs near: {}, {}'.format(tweet_lat, tweet_lon))
        nearby_pois = find_nearby_pois(geo_tweet_lat=tweet_lat, geo_tweet_lon=tweet_lon,
                                       poi_data=poi_dataframe, buffer_radius=search_radius,
                                       set_point_4326=True)
        poi_class_list = process_poi_names(poi_dataframe=nearby_pois)
        print('Found POI classes: {}'.format(poi_class_list))
        poi_text_list.append(' '.join(poi_class_list))
    print('Final POI text for analysis: {}'.format(poi_text_list))
    # Output the LDA topics based on POI class
    get_lda_model(poi_text_list,
                  grid_search_params=topic_modelling_search_params,
                  number_of_keywords=keyword_num,
                  keywords_file=keyword_file_name,
                  topic_predict_file=topic_predict_file_name,
                  saving_path=saving_path, grid_search_or_not=True,
                  topic_number=topic_number)
    return poi_text_list


def build_topic_model(df, colname, keyword_file_name, topic_number, topic_predict_file_name,
                      saving_path=topic_results):
    """
    :param df: the dataframe which contains the posted tweets
    :param colname: the column name we are interested
    :param keyword_file_name: the name of the saved file which contains the keyword for each topic
    :param topic_number: the number of topics we set for the topic modelling
    :param topic_predict_file_name: the name of the saved file which contains the topic prediction for
    each tweet
    :param saving_path: the saving path
    """
    text_list = list(df[colname])
    cleaned_text_list = [remove_non_alphabetical_letters(str(text)) for text in text_list]
    tokenized_text_list = [word_tokenize(text) for text in cleaned_text_list]
    bigram_phrases = gensim.models.phrases.Phrases(tokenized_text_list, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.phrases.Phrases(bigram_mod[tokenized_text_list])
    trigram_mod = gensim.models.phrases.Phraser(trigram_phrases)
    tokenized_text_joined = [' '.join(tweet_word_list) for tweet_word_list in tokenized_text_list]
    data_ready = process_words(tokenized_text_joined,
                               stop_words=unuseful_terms_set,
                               bigram_mod=bigram_mod, trigram_mod=trigram_mod)
    # np.save(os.path.join(read_data.desktop, 'saving_path', keyword_file_name[:-12]+'_text_topic.pkl'), data_ready)
    # Draw the distribution of the length of the tweet: waiting to be changed tomorrow
    data_sentence_in_one_list = [' '.join(text) for text in data_ready]
    # Get the median of number of phrases
    count_list = [len(tweet) for tweet in data_ready]
    print('The number of keywords we use is {}'.format(np.median(count_list)))
    get_lda_model(data_sentence_in_one_list,
                                     grid_search_params=topic_modelling_search_params,
                                     number_of_keywords=int(np.median(count_list)),
                                     keywords_file=keyword_file_name,
                                     topic_predict_file=topic_predict_file_name,
                                     saving_path=saving_path, grid_search_or_not=True,
                                     topic_number=topic_number)


def main_topics():
    """
    Main function ro run topic modeling before and during the Covid19
    :return:
    """
    tweets_before_covid = pd.read_csv(os.path.join(tweet_combined_path,
                                                   'hk_common_2018_translated.csv'),
                                      index_col=0, encoding='utf-8')
    tweets_during_covid = pd.read_csv(os.path.join(tweet_combined_path,
                                                   'hk_common_covid_translated.csv'),
                                      index_col=0, encoding='utf-8')
    build_topic_model(df=tweets_before_covid, colname='translate',
                      keyword_file_name='topics_before_covid.csv', topic_number=10,
                      topic_predict_file_name='topics_each_tweet_before_covid.csv')
    build_topic_model(df=tweets_during_covid, colname='translate',
                      keyword_file_name='topics_during_covid.csv', topic_number=10,
                      topic_predict_file_name='topics_each_tweet_during_covid.csv')


if __name__ == '__main__':
    # main_topics()
    # Load one geocoded tweet dataframe
    sample_tweets_before = gpd.read_file(os.path.join(
        records_path, 'user_cluster_tweets', 'tweets_shapefile', 'before_tweets_geo_cluster_3.shp'),
    encoding='utf-8', index_col=0)
    poi_data = gpd.read_file(os.path.join(shapefile_path, 'hk_poi.shp'), encoding='utf-8', index_col=0)
    nearby_poi_results = develop_lda_pois(coordinates_records=sample_tweets_before,
                                          poi_dataframe=poi_data,
                                          topic_number=10,
                                          keyword_file_name='poi_topics_cluster_3_before.csv',
                                          topic_predict_file_name='poi_topic_each_tweet_cluster_3_before.csv')
