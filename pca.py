from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from helpers import check_null_column, get_data_from_csv, check_same_value_column, init_parser


class PCA:
    """
    PCA is an implementation of principal component analysis
    """

    def __init__(self, desired_principal_components, path):
        self.path = path
        self.desired_principal_components = desired_principal_components
        self.extracted_eigenvectors = None
        self.headers, self.cols, self.rows = self._read_csv(path)
        self.feature_table = self.cols[2:]
        self.projected_data = 0

    @staticmethod
    def _read_csv(path):
        """
        reads the csv file given by path and returns headers, columns and rows
        columns represents features, rows represent the data points
        """
        headers, cols, rows = get_data_from_csv(path)
        print('# of cols before removal:', len(cols))
        for i, col in enumerate(cols):
            if i != 1 and check_null_column(col, percentage=80):  # pass class column
                print(f'feature {headers[i]} is removed, more than 80 percentage is null.')
                cols.pop(i)
                for row in rows:  # remove that feature from every row
                    row.pop(i)
            if check_same_value_column(col):
                print(f'feature {headers[i]} is removed due to same value.')
                cols.pop(i)
                for row in rows:  # remove that feature from every row
                    row.pop(i)
        print('# of cols after removal:', len(cols))
        return headers, cols, rows

    def _normalize(self):
        """
        normalize the features by subtracting mean and dividing by the standard deviation
        """
        feature_mean = []
        for i in range(len(self.feature_table)):
            feature_mean.append(0)
            sm = 0
            for value in self.feature_table[i]:
                if np.isnan(value):
                    value = 0.
                sm += value
            feature_mean[i] = sm / len(self.feature_table[i])
        feature_dev = []
        for i in range(len(self.feature_table)):
            feature_dev.append(0)
            sm = 0
            for value in self.feature_table[i]:
                if np.isnan(value):
                    value = 0.
                sm += ((value - feature_mean[i]) ** 2)
            feature_dev[i] = sqrt(sm / len(self.feature_table[i]))

        self.normalized_features = self.feature_table
        for i in range(len(self.feature_table)):
            for j in range(len(self.feature_table[i])):
                if np.isnan(self.feature_table[i][j]):
                    self.normalized_features[i][j] = 0.
                else:
                    self.normalized_features[i][j] = (self.feature_table[i][j] - feature_mean[i]) / feature_dev[i]
        self.normalized_features = np.array(self.normalized_features)

    def _fit(self):
        """
        normalize features and extract eigenvectors of covariance matrix
        """
        self._normalize()
        # calculate covariance matrix
        covariance_matrix = np.cov(self.normalized_features)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]
        # take first n eigenvectors with higher eigenvalues
        self.extracted_eigenvectors = eigenvectors[0:self.desired_principal_components]

    def plot_explained_variance(self):
        """
        plot explained variance against number of principal components
        """
        total_variance = sum(self.eigenvalues)
        explained_variances = []
        cumulative = [0]
        for i in range(len(self.eigenvalues)):
            explained_variances.append(self.eigenvalues[i] / total_variance)
            cumulative.append(cumulative[i] + (self.eigenvalues[i] / total_variance))
        cumulative.pop(0)
        plt.xlabel("Number of components")
        plt.ylabel("Explained variance")
        plt.bar(np.arange(len(self.eigenvalues)), explained_variances, align='center', label='individual')
        plt.step(np.arange(len(self.eigenvalues)), cumulative, where='mid', label='cumulative')
        plt.legend(loc='upper left')
        plt.savefig(f'plots/pca-explained-variance.png')
        print(f'Total variance explanied for all dimensions:')
        for i, val in enumerate(cumulative):
            print(f"{i+1}:, {val}")

    def save_data(self):
        """
        project the data with dot product and save array
        """
        self._fit()
        self.projected_data = np.dot(self.normalized_features.T, self.extracted_eigenvectors.T)
        if 'test' in self.path:
            np.save(f'saved/pca_data_test_dim{self.desired_principal_components}.npy', self.projected_data)
        else:
            np.save(f'saved/pca_data_dim{self.desired_principal_components}.npy', self.projected_data)

    def plot_projected(self, dim=2):
        """
        plot the projected data with colored class numbers
        """
        if dim == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(self.projected_data[:, 0], self.projected_data[:, 1], self.projected_data[:, 2],
                         c=self.cols[1])
            fig.savefig(f'plots/pca-plot-{dim}.png')
            fig.show()
        elif dim == 1:
            plt.scatter(self.projected_data[:, 0], self.cols[1])
            plt.savefig(f'plots/pca-plot-{dim}.png')
            plt.show()
        else:
            plt.scatter(self.projected_data[:, 0], self.projected_data[:, 1], c=self.cols[1])
            plt.savefig(f'plots/pca-plot-{dim}.png')
            plt.show()


if __name__ == '__main__':
    args = init_parser(args=[
        {
            'name': '--pca_dims',
            'dest': 'pca_dims',
            'default': 2,
            'choices': [1, 2, 3],
            'type': int,
            'help': 'Use PCA',
        },
    ])
    dim = args.pca_dims
    pca = PCA(desired_principal_components=dim, path='data/aps_failure_training_set.csv')
    pca.save_data()
    pca.plot_projected(dim=dim)
    pca.plot_explained_variance()
