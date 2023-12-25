import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

from helpers import read_csv_with_pandas

# pca_data = np.load("pca_data.npy")
# df = pd.DataFrame(data=data)

df = read_csv_with_pandas(path='data/aps_failure_training_set.csv')

# print(df.describe(include='all').T)
# print(df.dtypes)
sns.boxplot(y=df["cn_008"].values)
plt.title("Boxplot of cn_008")
plt.show()
plt.close()

# outlier detection
X = df.drop(labels=['class', 'id'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = IsolationForest(random_state=0)
clf.fit(X_train)
y_pred = clf.predict(X_test)
pred = pd.DataFrame({'pred': y_pred})
pred['y_pred'] = np.where(pred['pred'] == -1, 1, 0)
y_pred = pred['y_pred']
print("Precision:", precision_score(y_test, y_pred))

# target - feature relationship
corr = df.drop(labels=['class', 'id'], axis=1).apply(lambda x: x.corr(df['class']))
print('correlation of features and the class:\n', corr)
plt.title('Target/Feature Correlation')
plt.plot(corr.values, np.zeros_like(corr.values), 'x')
plt.savefig('plots/target-feature-relationship.png')
plt.close()
