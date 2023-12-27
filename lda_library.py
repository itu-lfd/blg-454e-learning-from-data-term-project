import numpy as np
import pandas as pd

from helpers import get_data_from_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


#headers, cols, rows = get_data_from_csv("data/aps_failure_training_set.csv")
df = pd.read_csv("raw_data/aps_failure_training_set.csv")

features = df.iloc[:, 2:]
features.replace('na', np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')  # You can use 'median', 'most_frequent', or a custom strategy
features_imputed = imputer.fit_transform(features)


# Extract the 'class' column
target = df['class']


# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the features
normalized_features = scaler.fit_transform(features_imputed)   

lda = LinearDiscriminantAnalysis(n_components=1)

# Fit the LDA model with the normalized features and target variable
lda.fit(normalized_features, target)

# Transform the features using the fitted LDA model
lda_transformed = lda.transform(normalized_features)

lda_df = pd.DataFrame(data=lda_transformed, columns=['LDA_Component_1'])
lda_df['id'] = df['id']
lda_df['class'] = df['class']


plt.figure(figsize=(10, 6))
colors = {'neg': 'red', 'pos': 'blue'}  # Adjust colors as needed
plt.scatter(lda_df['LDA_Component_1'],np.zeros_like(lda_df['LDA_Component_1']), c=lda_df['class'].map(colors), alpha=0.5)
plt.title('LDA-Transformed Features')
plt.xlabel('LDA_Component_1')
plt.ylabel('LDA_Component_2')
plt.show()
print(lda_df.head())


