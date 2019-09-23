import numpy as np
import pickle
import pandas as pd
import time
import csv

# from preprocessing import preprocessing

from stop_words import get_stop_words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from gensim.utils import simple_preprocess

def load_dataset(filename):
    start = time.time()
    print('Loading dataset...')

    data = pd.read_csv(filename)
    data.drop(columns='label_quality')
    data.drop_duplicates(subset ="title", 
                               keep = False,
                               inplace = True)
    
    print('Data shape: ', data.shape)
    end = time.time()
    print('Loaded! - Time:', round(end-start, 4))
    return data


def generate_subdataset(data, language):
    start = time.time()
    print('Generating ', language, ' dataset')

    data = data.loc[data['language'] == language]

    category = data['category']
    data = data['title']

    nCategory = 'data/category_pt.csv' if language=='portuguese' else 'data/category_sp.csv'
    nData = 'data/data_pt.csv' if language=='portuguese' else 'data/data_sp.csv'
    category.to_csv(nCategory, index=False, header=False)
    data.to_csv(nData, index=False, header=False)

    print('Data shape ', data.shape)
    print('Category shape ', category.shape)

    end = time.time()
    print('CSV files saved! - Time:', round(end-start, 4))


def clean_text(text):
    text = str(text)
    text2 = ''
    for token in simple_preprocess(text):
        if len(token) > 1:
            text2 += token + ' '

    return text2


def preprocessing_file(input, output, language):
    start = time.time()
    print('Preprocessing ', language, ' data')

    data = pd.read_csv(input, names=['title'])
    print('Initial data shape ', data.shape)

    text_corpus = []
    for sentence in data['title']:
        text_corpus.append(clean_text(sentence))
        
    data = pd.DataFrame(data=text_corpus, columns=['title'])

    data.to_csv(output, index=False, header=False)

    print('Final data shape ', data.shape)
    
    end = time.time()
    print('Preprocess completed! - Time:', round(end-start, 4))


# def preprocessing_data(input, output, preprocess=True):
#     start = time.time()
#     print('Preprocessing data')

#     if preprocess: preprocessing(input, output)

#     data = pd.read_csv(output, sep='\n', names=['title'])
#     print('Data shape ', data.shape)

#     # data['title'] = data['title'].str.replace("[0-9]", "")
#     data['title'] = data['title'].str.replace("[-,._/!]", "")
#     data['title'] = data['title'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
#     data.to_csv('data/pData.csv', index=False, header=False)

#     print('Data shape ', data.shape)
    
#     end = time.time()
#     print('Preprocess completed! - Time:', round(end-start, 4))


def token_counts(input, output, language):
    start = time.time()
    print('Converting a collection of ', language, ' text documents to a matrix of token counts')

    stop_words = get_stop_words(language)
    count_vec = CountVectorizer(stop_words=stop_words)

    data = pd.read_csv(input, names=['title'])
    data = data.replace(np.nan, "null", regex=True)
    data = data.replace("", "null", regex=True)
    print('Data shape ', data.shape)

    vectorizer = count_vec.fit_transform(data['title'])

    pickle.dump(vectorizer,open(output, 'wb'))
    
    end = time.time()
    print('Generated bag-of-words - Time:', round(end-start, 4))


def tfidf(input, output, language):
    start = time.time()
    print('Transforming a ', language, ' count matrix to a normalized tf-idf representation')
    
    tfidf_transformer = TfidfTransformer()
    loaded_vec = pickle.load(open(input, 'rb'))
    fit_tfidf = tfidf_transformer.fit_transform(loaded_vec)
    pickle.dump(fit_tfidf, open(output, 'wb'))

    end = time.time()
    print('TF-IDF transformer completed! - Time:', round(end-start, 4))


def train(input, output, csvFile, language):
    start = time.time()
    print('Training ', language, ' model')

    X = pickle.load(open(input, 'rb'))
    # X = np.array(loaded_tfidf, dtype='float16')
    # del loaded_tfidf

    y = pd.read_csv(csvFile, names=['category']).values.ravel()
    # y = np.array(y['category'], dtype='float16')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = [
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0)
    ]

    classifier = models[0]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    pickle.dump(classifier, output)

    end = time.time()
    print('Training completed! - ', round(end-start, 4))
