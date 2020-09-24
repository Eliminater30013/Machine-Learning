import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter

style.use('fivethirtyeight')

# df = pd.read_csv('breast-cancer-wisconsin.data')
# df.replace('?', -99999, inplace=True)  # clean the data
# df.drop(['id'], 1, inplace=True)  # id column is not useful so drop it and replace it with true
# X = np.array(df.drop(['class'], 1))  # define our features ( everything except id and class)
# #  drop the column (1) instead of a row(0)
# y = np.array(df['class'])  # and our label
# #2 - malignant 4 - benign
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)
# print(accuracy)
# example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 1, 8, 3, 2, 1]])  # any size of np array
# example_measures = example_measures.reshape(len(example_measures), -1)  # reshape it
# prediction = clf.predict(example_measures)  # predict it
# print(prediction)
# plot1 = [1,3]
# plot2 = [2,5]
# euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )
# print(euclidean_distance)


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k: # ensure the points are valid
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]] # returns a list that contains the 3 closest points
    vote_result = Counter(votes).most_common(1)[0][0] # give me the class name and mostcommnon(X) just returns the X number of elements
    # return most common distances
    return vote_result


dataset = {
    'k': [[1, 2], [2, 3], [3, 1]],
    'r': [[6, 5], [7, 7], [8, 6]]
}
new_features = [4, 5]
result = k_nearest_neighbors(dataset, new_features)
print(result)
for i in dataset: # loop through k and r which are colours!
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i) # different colour for each data

plt.scatter(new_features[0], new_features[1], s=100, color=result)
plt.show()
