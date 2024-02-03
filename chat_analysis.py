import pandas as pd
import re
import emoji
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from nltk import FreqDist
from nltk import bigrams, trigrams, pos_tag
from nltk.corpus import stopwords
from textstat.textstat import textstatistics 
from collections import Counter
import numpy as np

def tokenize_messages(message):
    tokens = word_tokenize(message)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

# Word Count
def word_count(message):
    return len(tokenize_messages(message))

# Use of Punctuation
def punctuation_count(message):
    return sum(1 for char in message if char in string.punctuation)

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
