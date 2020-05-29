import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# You don't necessarily have to use this
from sklearn.decomposition import PCA
# You don't necessarily have to use this
from sklearn.cluster import KMeans
# You don't necessarily have to use this
from sklearn.preprocessing import StandardScaler
from KMeansClustering import Kmeans
from sklearn import preprocessing

# Part of the following code was borrowed from the test code utilized by

df = pd.read_csv('titanic_copy.csv')
df.drop(['Name'], 1, inplace=True)
# print(df.head())
df.fillna(0, inplace=True)


def non_num(df):

    # Non-numerical values require to be changed for the model to perform preds
    # and clusters
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()
            # finding just the uniques
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x += 1
            # now we map the new "id" vlaue
            # to replace the string.
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = non_num(df)

X = np.array(df.drop(['Survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['Survived'])


colors = 10*["g", "r", "c", "b", "k"]

# Dumy data

data = np.array([[2, 6], [7, 9], [16, 17], [6, 9], [3, 6], [23, 15], [9, 2],
                 [6, 5], [1.2, 1.2], [1.5, 1.3]])

k = Kmeans()
k.fit(data)

for kmean in k.kmeans:
    plt.scatter(k.kmeans[kmean][0], k.kmeans[kmean][1],
                marker="o", color="k", s=150, linewidths=5)

for cluster in k.clusters:
    color = colors[cluster]
    for featureset in k.clusters[cluster]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color,
                    s=150, linewidths=5)

plt.show()

k = Kmeans()
k.fit(X)

# correct = 0
# for i in range(len(X)):
#
#     predict_me = np.array(X[i].astype(float))
#     predict_me = predict_me.reshape(-1, len(predict_me))
#     prediction = k.predict(predict_me)
#     if prediction == y[i]:
#         correct += 1
#
# print(f'Model prediction accuracy is: {correct/len(X)}')
