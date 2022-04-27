from typing import List

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from imblearn.under_sampling import TomekLinks

DataFrame = pd.DataFrame
Series = pd.Series


class CustomDataFrame(DataFrame):
    """
    Append custom methods and attributes to DataFrame class

    Methods:
        filter_by_column(column, search_text)
            Searches column and returns DataFrame where text matches search_text param
    """
    def filter_by_column(self, column:str, search_text:str) -> DataFrame:
        return self[self[column].str.contains(search_text)]
    

def series_to_wordcloud(column: Series)-> WordCloud:
    """transform pandas Series[text] to wordcloud object"""
    text = series_to_text(column)
    return WordCloud(width=800, height=400).generate(text)


    
def display_wordcloud(wordcloud: WordCloud) -> None:
    """Graphical display of wordcloud object"""
    fig, ax = plt.subplots(figsize=(20,20))
    
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.show()
    
    
def display_sequence(sequence:List, limit = 10) -> None:
    """Itterates through sequence, prints to user"""
    for index, content in enumerate(sequence):
        if index >= limit:
            return
        
        print(f'{index}:\n{content}')
        print('-------------------------------------------------------------------------------------------------------------------------------')


def series_to_text(series:Series) -> str:
    """Converts Series[str] to a single text output"""
    return ' '.join(series)


def extract_words(series:Series) -> List[str]:
    """Extract words from Series[str] object as list"""
    return series_to_text(series).split(' ')


def unique_words(series:Series) -> List[str]:
    """Returns unique words within Series[str] object as list"""
    return list(set(extract_words(series)))


# ---
# Specific to current data set (Twitter sentiment analysis):
def get_longest_tweets(df, length=5) -> DataFrame:
    """Returns sorted DataFrame from longest to shortest tweet"""
    df = df.copy(deep=True)
    df['tweet_length'] = [len(text) for text in df.message.values]
    return df.sort_values('tweet_length', ascending=False).iloc[:length].message


def prep_data(data):
    """Correction csv formating mistake as tokens and not strings"""
    data = data.copy(deep=True)
    data.message = [eval(t) for t in data.message.values]
    data.message = [' '.join(t) for t in data.message.values]
    return data


def resample_tomeklinks(X, y):
    """Applies undersampling technique, returns (X_resampled, y_resampled)"""
    return TomekLinks().fit_sample(X, y)


def submit_test(tweetid:List[int], predictions:List[int]) -> None:
    """DataFrame to Csv as specified for Kaggle submission"""
    submission = pd.DataFrame({'tweetid':tweetid,
                               'sentiment':predictions})
    submission.to_csv(input('Submission name and version: \n >>> ')+'.csv', index=False)