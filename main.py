import argparse
from productCategorization import (generate_subdataset,
                                   preprocessing_file,
                                   token_counts,
                                   tfidf,
                                   train)

def main(args):
    if args.gen_subdataset:
        generate_subdataset(load_dataset('data/train.csv'), 'portuguese')
        generate_subdataset(load_dataset('data/train.csv'), 'spanish')

    if args.preprocessing:
        preprocessing_file('data/data_pt.csv', 'data/output_pt.csv', 'portuguese')
        preprocessing_file('data/data_sp.csv', 'data/output_sp.csv', 'spanish')

    if args.token_counts:
        token_counts('data/output_pt.csv', 'data/feature_pt.pkl', 'portuguese')
        token_counts('data/output_sp.csv', 'data/feature_sp.pkl', 'spanish')

    if args.tfidf:
        tfidf('data/feature_pt.pkl', 'data/tfidf_pt.pkl', 'portuguese')
        tfidf('data/feature_sp.pkl', 'data/tfidf_sp.pkl', 'spanish')

    train('data/tfidf_pt.pkl', 'data/model_pt.pkl', 'data/category_pt.csv', 'portuguese', args.training_model)
    train('data/tfidf_sp.pkl', 'data/model_sp.pkl', 'data/category_sp.csv', 'spanish'args.training_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Category Classifier.')
    parser.add_argument('--gen-subdataset', action='store_false', default=True,
                        help='Split original dataset into portuguese and spanish dataset.')
    parser.add_argument('--preprocessing', action='store_false', default=True,
                        help='Apply the preprocessing function to datasets')
    parser.add_argument('--token-counts', action='store_false', default=True,
                        help='Converting a collection of text documents to a matrix of token counts')
    parser.add_argument('--tfidf', action='store_false', default=True,
                        help='Transforming a count matrix to a normalized tf-idf representation')
    parser.add_argument('--training-model', type=str, choices=['linearSVC', 'multinomialNB', 'logisticRegression'],
                        default='linearSVC', help='Classifier model')

    args = parser.parse_args()
    main(args)
