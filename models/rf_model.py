import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

from helpers import read_csv_with_pandas, init_parser

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score, accuracy_score, \
    precision_score, recall_score, PrecisionRecallDisplay, f1_score
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
        find the best hyperparameters for the classifier and save it with numpy
        """
        if os.path.exists(fname):
            best_params = np.load(fname, allow_pickle=True).item()
            print("Best hyperparameters: {}".format(best_params))
            self.model = CalibratedClassifierCV(estimator=RandomForestClassifier(**best_params), cv=3)
            return

        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=params,
            cv=3,
            return_train_score=True,
        )
        # Fit the random search object to the data
        # Create a variable for the best model
        self.model = CalibratedClassifierCV(estimator=rand_search.best_estimator_, cv=3)
        print('Best hyperparameters:', rand_search.best_params_)
        np.save(fname, rand_search.best_params_)

    def predict(self, threshold=0.5):
        """
        predict y values for the test dataset after fit and return predicted values
        """
        param_dist = {
            'n_estimators': [10, 25, 50, 80, 100],
            'max_depth': [5, 10, 20, 50],
            'min_samples_split': [2, 5, 10, 50],
        }
        if self.pca_dims:
            self.find_best(param_dist, fname=f'../saved/best_hyper_params_pca{pca_dims}.npy')
        else:
            self.find_best(param_dist, fname=f'../saved/best_hyper_params.npy')
        self.model.fit(self.X_train, self.y_train)
        predicted_prob = self.model.predict_proba(self.X_test)
        return predicted_prob, (predicted_prob[:, 1] >= threshold).astype('int')

    def save_test_file(self, X_test):
        y_pred = self.model.predict(X_test)
        d = {'id': [], 'class': []}
        for id, pred in enumerate(y_pred):
            d['id'].append(id+1)
            d['class'].append('pos') if pred else d['class'].append('neg')
        pd.DataFrame(data=d).to_csv('../submissions/rf_submission.csv', index=False)

    def plot(self, y_pred, y_prob):
        """
        plot confusion matrix and precision recall for predicted and real values
        """
        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.savefig('../plots/rf_confusion_matrix.png')
        PrecisionRecallDisplay.from_predictions(
            self.y_test, y_prob[:, 1], name="RandomForest", plot_chance_level=True,
        )
        plt.savefig('../plots/rf_precision_recall.png')

    def print_scores(self, y_pred, y_prob):
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        average_precision = average_precision_score(self.y_test, y_prob[:, 1])
        f1_scr = f1_score(self.y_test, y_pred, average='macro')
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Average Precision:", average_precision)
        print("Average macro f1 score:", f1_scr)
        print("Recall:", recall)


if __name__ == '__main__':
    args = init_parser(args=[  # take the number of principal components as argument
        {
            'name': '--pca_dims',
            'dest': 'pca_dims',
            'default': 2,
            'choices': [0, 1, 2, 3],
            'type': int,
            'help': 'Use PCA',
        },
    ])
    pca_dims = args.pca_dims
    df = read_csv_with_pandas(path='../data/aps_failure_training_set.csv')
    if pca_dims:
        print(f'PCA will be applied! Number of principal components is {pca_dims}.')
        pca_data = np.load(f'../saved/pca_data_dim{pca_dims}.npy')
        pca_data_test = np.load(f'../saved/pca_data_test_dim{pca_dims}.npy')
        X = pd.DataFrame(data=pca_data)
        X_test = pd.DataFrame(data=pca_data_test)
    else:
        print('PCA will not be applied!')
        X = df.drop(labels=['id', 'class'], axis=1)
        df_test = pd.read_csv('../data/aps_failure_test_set.csv')
        df_test.replace(to_replace='na', value=0, inplace=True)
        X_test = df_test.drop(labels=['id'], axis=1)
    y = df['class']
    rf = RandomForestModel(X, y, pca_dims)
    prob, pred = rf.predict(threshold=0.95)
    rf.plot(pred, prob)
    rf.print_scores(pred, prob)
    rf.save_test_file(X_test)
