# Mercado Livre Data Challenge 2019
The Mercado Livre Data Challenge is a Latin America competition to fit a model to categorize product's listing based on their titles. For more details see [MeLi Data Challenge](https://ml-challenge.mercadolibre.com/).

## Install
### Download the code
To download the code, run:
```
git clone https://github.com/luanagbmartins/ml-challenge.git
```

### Dependencies
You will need the following dependencies:
- python 3
- conda. (Optional for environment creation and management)

Create the environment from the environment.yml file:
```
conda env create -f environment.yml
```

### Download the training data
Download the data files from [MeLi Downloads](https://ml-challenge.mercadolibre.com/downloads). Place and extract the files in the following locations:
- data/train.csv
- data/test.csv

## Training

The training pipeline is divided by following steps:
- Generate Subdataset: Split original dataset into portuguese and spanish dataset.
- Preprocessing: Apply the preprocessing function to wich dataset.
- Token Counts: Converting a collection of text documents to a matrix of token counts.
- TF-IDF: Transforming a count matrix to a normalized TF-IDF representation.
- Train: Train a Classifier model. Choose between: linearSVC, multinomialNB, logisticRegression.
