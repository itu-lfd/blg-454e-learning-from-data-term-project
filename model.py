import os.path

from matplotlib import pyplot as plt

from helpers import read_csv_with_pandas

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pickle


class RandomForestModel:
    """
    Random forest classifier implementation for a given dataset
    """
    def __init__(self, path):
        self.model = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.df = read_csv_with_pandas(path=path)
        X = self.df.drop(labels=['id', 'class'], axis=1)
        y = self.df['class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def find_best(self, params):
        """
        find the best hyperparameters for the classifier and save it with pickle
        """
        if os.path.exists('best_hyper_params.pickle'):
            with open('best_hyper_params.pickle', 'rb') as handle:
                best_params = pickle.load(handle)
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
        with open('best_hyper_params.pickle', 'wb') as handle:
            pickle.dump(rand_search.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
        best_rf = self.find_best(param_dist)
        best_rf.fit(self.X_train, self.y_train)
        return best_rf.predict(self.X_test)

    def plot_confusion_matrix(self, y_pred):
        """
        plot confusion matrix for predicted and real values
        """
        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.savefig('plots/rf_confusion_matrix.png')
        plt.show()


rf = RandomForestModel(path='data/aps_failure_training_set.csv')
pred = rf.predict()
rf.plot_confusion_matrix(pred)
