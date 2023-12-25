import argparse
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from helpers import read_csv_with_pandas, init_parser

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score, accuracy_score, \
    precision_score, recall_score, PrecisionRecallDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split


class RandomForestModel:
    """
    Random forest classifier implementation for a given dataset
    """

    def __init__(self, X, y, pca_dims=0):
        self.model = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.pca_dims = pca_dims
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def find_best(self, params, fname):
        """
        find the best hyperparameters for the classifier and save it with pickle
        """
        if os.path.exists(fname):
            best_params = np.load(fname, allow_pickle=True).item()
            return RandomForestClassifier(**best_params)

        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=params,
            cv=3,
            return_train_score=True,
        )
        # Fit the random search object to the data
        rand_search.fit(self.X_train, self.y_train)
        # Create a variable for the best model
        best_rf = rand_search.best_estimator_
        print('Best hyperparameters:', rand_search.best_params_)
        np.save(fname, rand_search.best_params_)
        return best_rf

    def predict(self):
        """
        predict y values for the test dataset after fit and return predicted values
        """
        param_dist = {
            'n_estimators': [10, 25, 50, 80, 100],
            'max_depth': [5, 10, 20, 50],
            'min_samples_split': [2, 5, 10, 50],
        }
        if self.pca_dims:
            best_rf = self.find_best(param_dist, fname=f'saved/best_hyper_params_pca{pca_dims}.npy')
        else:
            best_rf = self.find_best(param_dist, fname=f'saved/best_hyper_params.npy')
        best_rf.fit(self.X_train, self.y_train)
        return best_rf.predict(self.X_test)

    def plot_confusion_matrix(self, y_pred):
        """
        plot confusion matrix for predicted and real values
        """
        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.savefig('plots/rf_confusion_matrix.png')
        PrecisionRecallDisplay.from_predictions(
            self.y_test, y_pred, name="RandomForest", plot_chance_level=True,
        )
        plt.savefig('plots/rf_precision_recall.png')

    def print_scores(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        average_precision = average_precision_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Average Precision:", average_precision)
        print("Recall:", recall)


if __name__ == '__main__':
    args = init_parser(args=[  # take the number of principal components as argument
        {
            'name': '--pca_dims',
            'dest': 'pca_dims',
            'default': 0,
            'choices': [0, 1, 2, 3],
            'type': int,
            'help': 'Use PCA',
        },
    ])
    pca_dims = args.pca_dims
    df = read_csv_with_pandas(path='data/aps_failure_training_set.csv')
    if pca_dims:
        print(f'PCA will be applied! Number of principal components is {pca_dims}.')
        pca_data = np.load(f'saved/pca_data_dim{pca_dims}.npy')
        X = pd.DataFrame(data=pca_data)
    else:
        print('PCA will not be applied!')
        X = df.drop(labels=['id', 'class'], axis=1)
    y = df['class']
    rf = RandomForestModel(X, y, pca_dims)
    pred = rf.predict()
    rf.plot_confusion_matrix(pred)
    rf.print_scores(pred)
