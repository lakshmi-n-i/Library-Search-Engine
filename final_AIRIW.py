''' import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import string
import gensim
import operator
import re
from flask import Flask, render_template, request
from gensim import corpora
import pickle


#df_books = pd.read_csv('Book_Dataset_1.csv')

app = Flask(__name__)

dataset = "Book_Dataset_1.csv"
df_books = pd.read_csv(dataset)

columns_to_remove = ['Price', 'Price_After_Tax', 'Tax_amount', 'Avilability', 'Number_of_reviews']
df_books = df_books.drop(columns=columns_to_remove)
from spacy.lang.en.stop_words import STOP_WORDS

spacy_nlp = spacy.load('en_core_web_sm')

#create list of punctuations and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

#function for data cleaning and processing
#This can be further enhanced by adding / removing reg-exps as desired.

def spacy_tokenizer(sentence):
    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    #remove unwanted lines starting from special charcters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    
    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)
    
    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    #creating token object
    tokens = spacy_nlp(sentence)
    
    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    
    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    
    #return tokens
    return tokens

df_books['Book_Description_tokenized'] = df_books['Book_Description'].map(lambda x: spacy_tokenizer(x))
book_plot = df_books['Book_Description_tokenized']



#creating term dictionary
dictionary = corpora.Dictionary(book_plot)

#list of few which which can be further removed
stoplist = set('hello and if this can would should could tell ask stop come go')
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

#print top 50 items from the dictionary with their unique token-id
dict_tokens = [[[dictionary[key], dictionary.token2id[dictionary[key]]] for key, value in dictionary.items() if key <= 50]]
corpus = [dictionary.doc2bow(desc) for desc in book_plot]
word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]
book_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
book_lsi_model = gensim.models.LsiModel(book_tfidf_model[corpus], id2word=dictionary, num_topics=300)
gensim.corpora.MmCorpus.serialize('book_tfidf_model_mm', book_tfidf_model[corpus])
gensim.corpora.MmCorpus.serialize('book_lsi_model_mm',book_lsi_model[book_tfidf_model[corpus]])

#Load the indexed corpus
book_tfidf_corpus = gensim.corpora.MmCorpus('book_tfidf_model_mm')
book_lsi_corpus = gensim.corpora.MmCorpus('book_lsi_model_mm')


from gensim.similarities import MatrixSimilarity
book_index = MatrixSimilarity(book_lsi_corpus, num_features = book_lsi_corpus.num_terms)

from operator import itemgetter

def search_similar_books(search_term):

    query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = book_tfidf_model[query_bow]
    query_lsi = book_lsi_model[query_tfidf]

    book_index.num_best = 5

    books_list = book_index[query_lsi]

    books_list.sort(key=itemgetter(1), reverse=True)
    book_names = []

    for j, book in enumerate(books_list):
      book_names.append (
          {
              'Relevance': round((book[1] * 100),2),
              'book Title': df_books['Title'][book[0]],
              'book Plot': df_books['Book_Description'][book[0]]
          }

        )
      if j == (book_index.num_best-1):
          break

    return pd.DataFrame(book_names, columns=['Relevance','book Title','book Plot'])

# search for book titles that are related to below search parameters
#this needs to be in the frontend so user can specify which type of books they want
print(search_similar_books('airport')) '''

import pickle

# Open the pickle file in read-binary mode
with open('models.pickle', 'rb') as f:
    # Load the objects from the pickle file
    objects = pickle.load(f)

# Print the objects to inspect their contents
print(objects)
