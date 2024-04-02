import pickle
import numpy as np

### Multi-constraint datasets

def load_nobel_city():
    filename = "./factual_queries/data/nobel_multiconstraint.pkl"
    items = pickle.load(open(filename, "rb"))
    return items

def load_word_startend():
    filename = "./factual_queries/data/word_multiconstraint.pkl"
    items = pickle.load(open(filename, "rb"))
    return items

def load_books():
    filename = "./factual_queries/data/books_multiconstraint.pkl"
    items = pickle.load(open(filename, "rb"))
    return items
