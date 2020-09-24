import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

#  Final K_Nearest Neighbours example


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    confidence = Counter(votes).most_common(1)[0][1] / k
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result, confidence


df = pd.read_csv('breast-cancer-wisconsin.data')  # Load in the data as a list
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
#print(df.head())
full_data = df.values.astype(float).tolist()  # ensure the lists are the floats
# print(full_data[:5])
random.shuffle(full_data)
test_size = 0.4
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]  # first 80% of the data as training, up to the last 20%
test_data = full_data[-int(test_size * len(full_data)):]  # last 20% as testing data, [36:], -Ve, from bottom
for i in train_data:
    train_set[i[-1]].append(i[:-1]) # go to 2,4 and put in the relevant data for each classification
for i in test_data:
    test_set[i[-1]].append(i[:-1]) # go to 2,4 and put in the relevant data for each classification
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print(vote, confidence)
        total += 1
print(correct, total)
print('Accuracy:', correct/total)