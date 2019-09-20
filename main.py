from productCategorization import *

if __name__ == '__main__':
    generate_subdataset(load_dataset('data/train.csv'), 'portuguese')
    generate_subdataset(load_dataset('data/train.csv'), 'spanish')

    preprocessing_file('data/data_pt.csv', 'data/output_pt.csv', 'portuguese')
    preprocessing_file('data/data_sp.csv', 'data/output_sp.csv', 'spanish')

    token_counts('data/output_pt.csv', 'data/feature_pt.pkl', 'portuguese')
    token_counts('data/output_sp.csv', 'data/feature_sp.pkl', 'spanish')

    tfidf('data/feature_pt.pkl', 'data/tfidf_pt.pkl', 'portuguese')
    tfidf('data/feature_sp.pkl', 'data/tfidf_sp.pkl', 'spanish')

    train('data/tfidf_pt.pkl', 'data/model_pt.pkl', 'data/category_pt.csv', 'portuguese')
    train('data/tfidf_sp.pkl', 'data/model_sp.pkl', 'data/category_sp.csv', 'spanish')