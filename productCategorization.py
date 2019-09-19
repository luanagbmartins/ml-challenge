import numpy as np
from preprocessing import preprocessing
import pickle
import pandas as pd
from stop_words import get_stop_words
from gensim import corpora
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def remove_stopwords(rev, stop_words):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

if __name__ == '__main__':

    print('Loading dataset...')

    # file = 'data/train.csv'
    # data_train = pd.read_csv(file)

    # data_train.drop(columns='label_quality')
    # data_train.drop_duplicates(subset ="title", 
    #                            keep = False,
    #                            inplace = True)

    ###################################

    print('Generating portuguese dataset')

    # data_pt = data_train.loc[data_train['language'] == 'portuguese']

    # category_pt = data_pt['category']
    # data_pt = data_pt['title']

    # category_pt.to_csv(r'category_pt.csv', index=False, header=False)
    # data_pt.to_csv(r'data_pt.csv', index=False, header=False)

    # del category_pt
    # del data_pt

    ###################################

    print('Generating portuguese dataset')

    # data_sp = data_train.loc[data_train['language'] == 'spanish']

    # category_sp = data_sp['category']
    # data_sp = data_sp['title']

    # category_sp.to_csv(r'category_sp.csv', index=False, header=False)
    # data_sp.to_csv(r'data_sp.csv', index=False, header=False)

    # del category_sp
    # del data_sp

    ###################################

    # del data_train

    ###################################

    print('Preprocessing portuguese data')

    # preprocessing('data_pt.csv', 'output_pt.csv')

    data_pt = pd.read_csv('output_pt.csv', sep='\n', names=['title'])
    data_pt['title'] = data_pt['title'].str.replace("[^a-zA-Z#]", "")
    data_pt['title'] = data_pt['title'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    stop_words = get_stop_words('portuguese')
    data_pt = [remove_stopwords(r.split(), stop_words) for r in data_pt['title']]
    data_pt = pd.Series(data_pt).apply(lambda x: x.split())

    print('Generating portuguese dictionary')

    dictionary = corpora.Dictionary(data_pt)
    dictionary.filter_extremes(no_below=20, no_above=0.8)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in data_pt]
    dictionary.save_as_text('dictionary.txt')

    # del data_pt
    # del dictionary

    ###################################

    print('Portuguese TF-IDF')
 
    from gensim.models import TfidfModel
    tfidf = TfidfModel(dictionary=dictionary, normalize=True)
    xTrain = [tfidf[dictionary.doc2bow(doc)] for doc in data_pt]
    tfidf.save('tfidf.pkl')
 
    print(len(xTrain))

    del data_pt
    del dictionary

    ###################################

    print('Training portuguese model')

    loaded_category = pd.read_csv('category_pt.csv', names=['category'])
    yTrain = np.array(loaded_category['category']).flatten()
    clf = MultinomialNB().fit(xTrain, yTrain)
    pickle.dump(clf, open('model_pt.sav', 'wb')) 

    del xTrain
    del loaded_category
    del clf

    # ###################################

    # print('Preprocessing spanish data')

    # preprocessing('data_sp.csv', 'output_sp.csv')

    # data_sp = pd.read_csv('output_sp.csv')
    # data_sp['title'] = data_sp['title'].str.replace("[^a-zA-Z#]", "")
    # data_sp['title'] = data_sp['title'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    # data_sp = pd.Series(data_sp).apply(lambda x: x.split())

    # print('Generating spanish dictionary')

    # dictionary = corpora.Dictionary(data_sp)
    # dictionary.filter_extremes(no_below=20, no_above=0.8)
    # doc_term_matrix = [dictionary.doc2bow(rev) for rev in data_sp]

    # del data_sp
    # del dictionary

    # ###################################

    # print('Spanish TF-IDF')

    # xTrain = TfidfTransformer().fit_transform(doc_term_matrix)

    # ###################################

    # print('Training spanish model')

    # loaded_category = pd.read_csv('category_sp.csv')
    # clf = MultinomialNB().fit(xTrain, loaded_category)
    # pickle.dump(clf, open('model_sp.sav', 'wb')) 

    # del xTrain
    # del loaded_category
    # del clf
