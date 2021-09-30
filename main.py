import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from fuzzywuzzy.fuzz import ratio
from difflib import SequenceMatcher
import numpy as np

# const definition


WELFARE = {1: "Rich", 2: "Poor"}

filename = "Boston_housing.data"
df = pd.read_table(filename, sep=' ').interpolate(method='cubic')
medv = df.values[:, 13]

medmedv = np.median(medv)

df["x15"] = medv < medmedv  # 0 -- меньше
u = df.select_dtypes(bool)
df[u.columns] = u.astype(int)
fd = df
test = 0.3
X = fd.values[:, :-1]
y = fd.values[:, -1]
# print(X)
# print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test)

# X_train = X[1::2, :]
# X_test = X[::2, :]
# y_train = y[1::2]
# y_test = y[::2]


acc = []

for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    res = pd.DataFrame()
    res['test'] = y_test
    res['pred'] = y_pred
    data_crosstab = pd.crosstab(res['test'],
                                res['pred'],
                                margins=False)

    acc.append(metrics.accuracy_score(y_test, y_pred))
    print(data_crosstab)

plt.plot(range(1, 50), acc)
plt.show()
