import re
from nltk import ngrams
from collections import Counter



def filter_ngram_repeating(text, n):
    words = re.findall(r'\w+', text.lower())
    
    n_grams = list(ngrams(words, n))
    n_gram_counts = Counter(n_grams)
    repeating_n_grams = [n_gram for n_gram, count in n_gram_counts.items() if count > 1]
    
    return repeating_n_grams