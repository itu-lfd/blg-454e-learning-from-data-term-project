import numpy as np
import sys, itertools
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.impute import SimpleImputer


from helpers import get_data_from_csv, impute_with_mean


class LDA:
    def __init__(self, n_components):
        self.linear_discriminants = None
        self.n_components=n_components

    @staticmethod
    def _read_csv(path):

        headers, cols, rows = get_data_from_csv(path)

        replaced_cols = []
        replaced_rows = []
        for j in range(len(headers)):
            replaced_col = [np.nan if val == 'na' else val for val in cols[j]]
            replaced_cols.append(replaced_col)
        
        for row in rows:
            replaced_row = [np.nan if val == 'na' else val for val in row]
            replaced_rows.append(replaced_row)
        return headers, replaced_cols, replaced_rows

    def take_features_target(self, headers, cols, rows):
        feature_headers = [headers[2:] for header in headers]
        features = [row[2:] for row in rows]

        target = cols[1]
        return features, target
    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))
            #print(SW)
            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        #print(np.linalg.det(SW))
        alpha = 0.1
        identity_matrix = np.identity(SW.shape[0])
        SW = SW + alpha * identity_matrix
        #print(np.linalg.det(SW))
        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)
"""
if __name__ =='__main__':
    lda = LDA(1)
    # df = pd.read_csv("raw_data/aps_failure_training_set.csv")
    headers, cols, rows = lda._read_csv("raw_data/aps_failure_training_set.csv")

    features, target = lda.take_features_target(headers, cols, rows)

    print(headers)
    # features = df.iloc[:, 2:]
    # features.replace(['na', 'NaN'], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')  # You can use 'median', 'most_frequent', or a custom strategy
    features_imputed = imputer.fit_transform(features)
    
    features_imputed = np.array(features_imputed)
    print(features_imputed.shape)
    # features_imputed = impute_with_mean(features)

    # print(df['id'])

    # target = df['class']


    print(features_imputed.shape)
    
    lda.fit(features_imputed, target)
    X_train_modified = lda.transform(features_imputed)
    target = np.array(target)

    # Assuming X_train_modified is a one-dimensional array
    plt.scatter(X_train_modified, np.zeros_like(X_train_modified), c=target, cmap='viridis', alpha=0.5)
    plt.xlabel('LDA Component 1')
    plt.title('Scatter Plot of LDA-transformed Data')
    plt.show()
"""