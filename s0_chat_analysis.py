import pandas as pd
import numpy as np
import re
import configparser
import json
from openai import OpenAI
from mistralai import Mistral

#import emoji
#from django.conf import settings
#settings.configure(DEBUG=True)

# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# from nltk.tokenize import word_tokenize, sent_tokenize
# import string
# from nltk import FreqDist
# from nltk import bigrams, trigrams, pos_tag
# from nltk.corpus import stopwords

from textstat.textstat import textstatistics 
from collections import Counter
from tqdm import tqdm



def get_five_shots(df):
    # Get random five messages from each session
    df_five_shots = df.groupby('sessionId').apply(lambda x: x.sample(5)).reset_index(drop=True)
    df_five_shots = df_five_shots[['sessionId','messageID','timestamp','content','rewritten']]
    # rename content -> original, rewritten -> neutral
    df_five_shots = df_five_shots.rename(columns={'content':'original','rewritten':'neutral','sessionId':'username'})
    df_five_shots.to_csv('f1_processed_user_chat_data/five_shots.csv')
    return df_five_shots


def parse_log_chats(filename):
    df = pd.read_csv('00_input_data/'+filename, sep='\t', header=None)
    df.columns = ['timestamp', 'log_level', 'message']
    # from the message column, filter only those which json has a field called fromUserId
    df_chats = df[df['message'].str.contains('userId') & df['message'].str.contains('"log"') & df['message'].str.contains('"content')]
    df_chats['content'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['content'])
    df_chats['fromUserId'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['userId'])
    df_chats['toUserId'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['toUserId'])
    df_chats['timestamp'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['timestamp'])
    df_chats['sessionId'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['sessionId'])
    df_chats['word_count'] = df_chats['content'].apply(lambda x: len(x.split()))
    # add messageID column to df_chats, first grouping the df_chats by sessionId and then adding a column with the index
    df_chats['messageID'] = df_chats.groupby('sessionId').cumcount()

    return df_chats


def postprocess_text(df):
    # Remove group chat duplicates
    df = remove_group_chat_duplicates(df)
    # First group by sessionID, then calculate the average_sentence_length and keep only the rows with word_count > average_sentence_length
    print(df.shape)
    df['average_sentence_length'] = df.groupby('sessionId')['word_count'].transform('median')
    print(df.shape)
    df = df[df['word_count'] >= df['average_sentence_length']]
    print(df.shape)
    # drop the average_sentence_length column
    df = df.drop(columns=['average_sentence_length'])
    print(df.shape)
   
    return df


def remove_group_chat_duplicates(df):
    # Step 1: Count the occurrences of each unique content
    content_counts = df['content'].value_counts()

    # Step 2: Identify content that appears at least three times
    content_to_keep = content_counts[content_counts >= 3].index

    # Step 3: Filter out rows and keep only the first occurrence of content that appears at least three times
    mask = df['content'].isin(content_to_keep)
    df_filtered = df[~mask | (mask & ~df.duplicated(subset='content', keep='first'))]

    return df_filtered


def tokenize_messages(message):
    tokens = word_tokenize(message)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

# Word Count
def word_count(message):
    return len(tokenize_messages(message))

def extract_functional_words(message):
    nlp = spacy.load("en_core_web_sm")
    
    # Process the input string using SpaCy
    doc = nlp(message)
    
    # Extract functional words (e.g., articles, prepositions, conjunctions)
    functional_words = [token.text for token in doc if token.pos_ in ['DET', 'ADP', 'CONJ']]
    
    return functional_words

def extract_adjectives(text):
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text using SpaCy
    doc = nlp(text)
    
    # Extract adjectives from the processed text
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    
    return adjectives



# Use of Punctuation
def punctuation_count(message):
    return sum(1 for char in message if char in string.punctuation)

def extract_punctuation(message):
    # Get all punctuation characters
    all_punctuation = string.punctuation
    
    # Extract punctuation from the message
    punctuation_in_message = [char for char in message if char in all_punctuation]
    
    return punctuation_in_message

# TTR is a measure of lexical richness that compares the number of unique words (types) to 
# the total number of words (tokens) in the text.
# Unlike vocabulary diversity, which is calculated per message,
# TTR can be calculated over varying lengths of text and is normalized for text length.
# def type_token_ratio(message):
#     words = nltk.word_tokenize(message)
#     return len(set(words)) / len(words) if words else 0

# Compute readability indices like Flesch Reading Ease or Gunning Fog Index to assess the complexity of the text.
# These scores can indicate how accessible or challenging the text is for readers.
def readability_score(message):
    return round(textstatistics().flesch_reading_ease(' '.join(message)), 2)

# Lexical density is the proportion of content words (nouns, verbs, adjectives, adverbs) in the text
# compared to the total number of words. A higher lexical density might indicate a more content-focused or formal style
def lexical_density(message):
    content_pos = {'NN', 'VB', 'JJ', 'RB'}  # Nouns, Verbs, Adjectives, Adverbs
    words = nltk.word_tokenize(message)
    tags = pos_tag(words)
    content_words = sum(1 for word, tag in tags if tag in content_pos)
    return content_words / len(words) if words else 0


# Applied on the whole message corpus
# Vocabulary Diversity: This function calculates the ratio of unique words to total words in the aggregated text.
# It first tokenizes the text into words, converts them to lowercase for standardization,
# and then calculates the ratio.
def vocabulary_diversity(corpus):
    words = word_tokenize(corpus)
    words = [word.lower() for word in words if word.isalpha()]
    return len(set(words)) / len(words) if words else 0
    
# Average Sentence Length: This function calculates the average number of words per sentence in
# the aggregated text. It tokenizes the text into sentences and then counts the words in each sentence.
def average_sentence_length(corpus):
    sentences = sent_tokenize(corpus)
    if len(sentences) == 0:
        return 0
    return sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)

# Analyze the frequency of bigrams (pairs of words) or trigrams (triplets of words). 
# This can reveal common phrases or topics in the chat data.
# N-grams can be particularly insightful for identifying colloquial expressions or recurring themes.
def ngram_frequency(corpus, n=2):
    # Tokenize the corpus and generate n-grams
    words = nltk.word_tokenize(corpus)
    if n == 2:
        ngrams = list(bigrams(words))
    elif n == 3:
        ngrams = list(trigrams(words))
    else:
        raise ValueError("n should be 2 for bigrams or 3 for trigrams")

    # Calculate frequency distribution
    ngram_freq = Counter(ngrams)
    return ngram_freq
    
# Function to calculate bigram frequency for each message
def calculate_bigram_frequency(message):
    return corpus_bigram_freq

# Function to calculate trigram frequency for each message
def calculate_trigram_frequency(message):
    return corpus_trigram_freq
    
def pos_distribution(corpus):
    # Tokenize the corpus and get POS tags
    words = nltk.word_tokenize(corpus)
    pos_tags = nltk.pos_tag(words)

    # Calculate frequency distribution of POS tags
    pos_freq = FreqDist(tag for (word, tag) in pos_tags)
    tag = []
    freq = []
    for f in pos_freq:
        tag.append(f)
        freq.append(pos_freq[f])

    df_pos_distribution = pd.DataFrame({
        'pos_tag': tag,
        'pos_freq': freq,
    })

    
    return df_pos_distribution

def word_length_distribution(corpus):
    # Tokenize the corpus and get word lengths
    words = nltk.word_tokenize(corpus)
    word_lengths = [len(word) for word in words if word.isalpha()]

    # Calculate frequency distribution of word lengths
    length_freq = FreqDist(word_lengths)
    word_length = []
    frequency = []
    for k in length_freq:
        word_length.append(k)
        frequency.append(length_freq[k])


    df_word_length_distribution = pd.DataFrame({
        'word_length': word_length,
        'frequency': frequency,
    })

    
    return df_word_length_distribution
    
# The term "stop word frequency" refers to the distribution of stop words in a text or corpu
def stop_word_frequency(corpus):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(corpus)
    total_words = len(words)
    stop_words_count = sum(1 for word in words if word.lower() in stop_words)

    return stop_words_count / total_words if total_words > 0 else 0
    
def get_top_ngrams(corpus, n=2, min_frequency=3):
    # Tokenize the corpus into words
    words = word_tokenize(corpus)
    
    # Exclude specific punctuation (e.g., ".")
    filtered_words = [word.lower() for word in words if word.isalpha() and word != "."]

    # Generate n-grams
    ngrams_list = list(bigrams(words)) if n == 2 else []

    # Calculate frequency distribution of n-grams
    ngram_freq = Counter(ngrams_list)

    # Filter and print n-grams with frequency greater than min_frequency
    top_ngrams = [(ngram, count) for ngram, count in ngram_freq.items() if count > min_frequency]
    ngrams = []
    frequency = []
    for ngram, count in top_ngrams:
        ngrams.append(ngram)
        frequency.append(count)

    # Creating a DataFrame
    df_grams = pd.DataFrame({
        'grams': ngrams,
        'frequency': frequency,

    })

    return df_grams
def get_top_words_by_pos(corpus):
    # Tokenize the corpus into words
    words = word_tokenize(corpus)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    # Perform POS tagging
    pos_tags = pos_tag(filtered_words)

    # Separate words by POS category
    verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
    adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
    nouns = [word for word, pos in pos_tags if pos.startswith('NN')]

    # Count occurrences of each category
    verb_counts = Counter(verbs)
    adjective_counts = Counter(adjectives)
    noun_counts = Counter(nouns)

    # Get the top 10 used words for each category along with frequency
    top_verbs = [(word, count) for word, count in verb_counts.most_common(10)]
    top_adjectives = [(word, count) for word, count in adjective_counts.most_common(10)]
    top_nouns = [(word, count) for word, count in noun_counts.most_common(10)]
    df_top_pos = pd.DataFrame({
        'verbs': top_verbs,
        'adjectives': top_adjectives,
        'nouns': top_nouns,
    })

    
    return df_top_pos
        
def extract_emojis(s):
    # Extract emojis using the emoji package
    all_emojis = ''.join(c for c in s if c in emoji.EMOJI_DATA)
    
    # Regex pattern for keyboard emoticons
    emoticon_pattern = re.compile(r'(:\)|;\)|:\(|:\D|:P|:O|:\||>:O|:\/|:\[|:\]|:\{|:\}|<3)')

    # Find all emoticons in the string
    found_emoticons = emoticon_pattern.findall(s)
    all_emoticons = ''.join(found_emoticons)

    # Combine emojis and emoticons
    all_emojis_emoticons = all_emojis + all_emoticons

    # Remove emojis and emoticons from the original message
    # Create a pattern that matches all found emojis and emoticons
    combined_pattern = re.compile('|'.join(re.escape(c) for c in all_emojis) + '|' + emoticon_pattern.pattern)
    
    cleaned_message = combined_pattern.sub(r'', s)

    # Return a tuple of cleaned message and emojis/emoticons
    return cleaned_message, all_emojis_emoticons or np.nan  # Returns 'N/A' if none found
