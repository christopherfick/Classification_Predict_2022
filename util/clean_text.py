import pandas as pd

import textblob
from textblob import TextBlob

from typing import List

import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

if not nltk.download('stopwords'):
    nltk.download('stopwords')
if not nltk.download('brown'):
    nltk.download('brown')
    
from collections import Counter

class Slanger(str):
    """Handling slang words in text objects can be treated as strings"""
    def __init__(self, text):
        self.text = text
        self.slang_dict = {'&amp': 'and',
                           'afaic': "as far as i'm concerned",
                           'afaik': 'as far as i know',
                           'afair': 'as far as i recall',
                           'afk': 'away from keyboard',
                           'asap': 'as soon as possible',
                           'bbl': 'be back later',
                           'bbs': 'be back soon',
                           'bfd': 'big fucking deal',
                           'bk': 'back',
                           'brb': 'be right back',
                           'btw': 'by the way',
                           'b2b': 'business to business',
                           'cya': 'goodbye',
                           'cu': 'goodbye',
                           'faq': 'frequently asked question',
                           'ffs': "for fuck's sake",
                           'fyi': 'for your information',
                           'gagf': 'go and get fucked',
                           'gg': 'good going',
                           'gj': 'good job',
                           'gl': 'good luck',
                           'hth': 'hope this helps',
                           'ianal': 'i am not a lawyer',
                           'ianars': 'i am not a rocket scientist',
                           'ic': 'i see',
                           'icydk': "in case you didn't know",
                           'icydn': "in case you didn't know",
                           'icudk': "in case you didn't know",
                           'iirc': 'if i recall correctly',
                           'imho': 'in my humble opinion',
                           'imo': 'in my opinion',
                           'imnsho': 'in my not so humble opinion',
                           'irc': 'internet relay chat',
                           'irl ," ""in real life""",'
                           'istr': 'i seem to recall',
                           'iydmma': "if you don't mind me asking",
                           'jj': 'just joking',
                           'jk': 'just kidding',
                           'jooc': 'just out of curiosity',
                           'k': 'ok',
                           'l8': 'late',
                           'l8r': 'later',
                           'liek': 'like',
                           'lmao': 'laughing',
                           'lol': 'laughing',
                           'myob': 'mind your own business',
                           'nm': 'not much',
                           'nvm': 'never mind',
                           'noyb': 'none of your business',
                           'np': 'no problem',
                           'o': 'oh',
                           'oic': 'oh i see',
                           'omg': 'oh my god',
                           'omfg': 'oh my fucking god',
                           'omfl': 'oh my fucking lag',
                           'ooc': 'out of character',
                           'ot': 'off topic',
                           'otoh': 'on the other hand',
                           'pfo': 'please fuck off',
                           'pita': 'pain in the ass',
                           "po (po'd)": 'piss off',
                           "po'd": 'pissed off',
                           'prog': 'computer program',
                           'prolly': 'probably',
                           'plz': 'please',
                           'pwn': 'dominate',
                           'p2p': 'person to person',
                           'qoolz': 'cool',
                           'r': 'are',
                           'rl': 'real life',
                           'rotfl': 'laughing',
                           'roflmao (or )': 'laughing',
                           'rotflmao': 'laughing',
                           'rtfa': 'read the fucking article ',
                           'rtfm': 'read the fucking manual',
                           'ru': 'are you',
                           'r8': 'right',
                           'sfw': 'safe for work',
                           'stfu': 'shut the fuck up',
                           'tbh': 'to be honest',
                           'thx': 'thanks ',
                           'ttfn': 'bye',
                           'tia': 'thanks in advance',
                           'ttyl': 'talk to you later',
                           'u': 'you',
                           'ur': "you're",
                           'w/e': 'whatever',
                           'w/o': 'without',
                           'wduwta': 'what do you want to talk about',
                           'wtf': 'what the fuck',
                           'woot': 'hell ye',
                           'w8': 'wait',
                           '<3': 'love',
                           '2b': 'to be'}


    # Should consider using a word tokenizer instead of spliting on spaces to catch words like 'g2g...'
    def correct(self):
        """If slang word found, translate to english equivalent"""
        return ' '.join([self.slang_dict.get(word)
                        if word in self.slang_dict
                        else word
                        for word in self.text.split()])

assert Slanger('cya l8r lol').correct() == 'goodbye later laughing'
assert Slanger('wtf is wrong with you').correct() == 'what the fuck is wrong with you'
assert Slanger('gl my friend <3').correct() == 'good luck my friend love'


def correct_spelling(text_tokens:List[str]) -> List[str]:
    """Corrects basic spelling errors. Note: Incorrectly classifies lots of words like 'epa' -> 'pa', explicitly added to grammer dictionary"""
    new_words = {'trump':1,
                 'epa':1}

    textblob.en.spelling.update(new_words)
    return [str(TextBlob(word).correct()) for word in text_tokens]

assert correct_spelling(['aligator', 'crockodile', 'lizard']) == ['agitator', 'crockodile', 'wizard']


def extract_url(text:str) -> List[str]:
    """Extract urls from text and return in the form ['url1', 'url2', 'url3', etc...]"""
    return re.findall("(?P<url>https?://[^\s]+)", text)

expected_results = ['https://athena.explore-datascience.net,', 'https://explore-datascience.net']
assert extract_url('Link to DataScience material: https://athena.explore-datascience.net, https://explore-datascience.net') == expected_results
                                                                

def remove_url(text:str) -> str:
    """Removes urls from text"""
    uncleaned_text = re.sub("(?P<url>https?://[^\s]+)", '', text)  #Text contains two white spaces where url once was
    return re.sub(' +', ' ', uncleaned_text)

assert remove_url('Lets learn DataScience: https://explore-datascience.net') == 'Lets learn DataScience: '


def remove_punctuation(text:str) -> str:
    return ''.join([l for l in text if l not in string.punctuation])

assert remove_punctuation('!@#$%? Dude!!! get a grip.') == ' Dude get a grip'  #Note space before the word 'Dude'


def stemm_text(tweet:str) -> str:
    """Stems text using SnowballStemmer"""
    tweet_tokenized = word_tokenize(tweet)
    return ' '.join([SnowballStemmer(language='english').stem(word) for word in tweet_tokenized])

assert stemm_text('greendays, greenlands, greentrees, greenie, greeny') == 'greenday , greenland , greentre , greeni , greeni'


def lemmatize_text(tweet:str) -> str:
    """Apply lemmatization text using WordNetLemmatizer"""
    tweet_tokenized = word_tokenize(tweet)
    return ' '.join([WordNetLemmatizer().lemmatize(word) for word in tweet_tokenized])

assert lemmatize_text('pigs dogs cats') == 'pig dog cat'


def remove_stopwords(text:str) -> str:
    """Removes stop words from text using dictionary for faster lookup times"""
    stopwords_dict = Counter(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stopwords_dict])

assert remove_stopwords('this is a great sentance, it is long and full of rich text') == 'great sentance, long full rich text'
assert remove_stopwords('this random, has order not in it.') == 'random, order it.'


def remove_words(text:str, words_to_remove:List[str]) -> str:
    """Removes words from text -> returns edited text"""
    return ' '.join([word
                    for word in text.split()
                    if word not in words_to_remove])

assert remove_words('there is a_ Firefly _in my boot...', ['Firefly']) == 'there is a_ _in my boot...'
assert remove_words('she sells sea shells', ['sea', 'shells']) == 'she sells'
assert remove_words('ping pong', ['ping pong']) == 'ping pong'


def quick_text_clean(text:str) -> str:
    """
    Filters and cleans twitter text: 
        Processess = [remove_urls, lower text, strip text, 
                    remove @users and #tags, correct slang, drop non ascii encodings, 
                    remove punctuations, remove numbers, drop tweet labels like rt->re tweet
                    and remove extra spaces between words]
    Params:
        text(str): The text to be filterd and cleaned
    
    Returns:
        text(str): The text Processed and cleaned

    Examples:
        >>> quick_text_clean("RT @NelsonMandela Everything seems impossible until it's done Ã°Å¸â€ #QuotesForLife")
        "everything seems impossible until its done" 
    """
    assert isinstance(text, str), f'Expected object of type string but got {type(text)} instead'
    text = remove_url(text)
    text = text.lower()
    text = text.strip()
    
    #Remove @users and #hashtags
    text = re.sub("@[A-Za-z0-9_]+","", text)
    text = re.sub("#[A-Za-z0-9_]+","", text)
    
    
    text = Slanger(text).correct()
    text = text.encode('utf-8','ignore').decode('ascii', 'ignore')  #Strange values in text 'Ã°Å¸â€...'
    text = text.replace('\n', '')
    
    text = remove_punctuation(text)
    
    text = re.sub('\w*\d\w*', ' ', text)  #Remove numbers from text
    
    # Remove tweet labels
    tweet_labels = ['rt', 'mt', 'dm']
    if text[0:2] in tweet_labels:
        text = text[3:]  
    
    # Remove double space between words
    text = re.sub(' +', ' ', text)
    
    return text.strip()

assert quick_text_clean("RT @NelsonMandela Everything seems impossible until it's done Ã°Å¸â€ #QuotesForLife") == "everything seems impossible until its done"
assert quick_text_clean("LOL ru afk ?") == 'laughing are you away from keyboard'
assert quick_text_clean('') == ''


def clean_text(text):
    """Correct grammer & remove stopwords"""
    text = quick_text_clean(text)
    text = remove_stopwords(text)
    text_tokens = word_tokenize(text)
    text_tokens = correct_spelling(text_tokens)
    return ' '.join(text_tokens)

assert clean_text('Thiss is a sample Texxt, that has been incorectly typed lol') == 'this sample text incorrectly type laughing'
assert clean_text('Gj man! you can alwayz do it') == 'good job man always'