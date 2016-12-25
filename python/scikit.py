import glob
import numpy as np
from sklearn import preprocessing
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier


def is_entertainment(category):
    entertainment = ['adult', 'arts', 'computers', 'games', 'home', 'society', 'shopping','sports']
    # entertainment = ['games']
    return 1 if category in entertainment else 0


vecs = {}

for line in open("model.txt", encoding="utf-8"):
    splitted_line = line.strip("\n").split(" ")
    word = splitted_line[0]
    vec = np.array([float(x) for x in splitted_line[1:]])
    vecs[word] = vec

categories = []

for cats in glob.glob("data/train/*"):
    category = cats.split("\\")[-1]
    categories.append(category)

X = []
y = []

for cat in categories:
    for filename in glob.glob("data/train/" + cat + "/*.txt"):
        file_vecs = []
        for line in open(filename, encoding="utf-8"):
            for word in line.split(" "):
                if word and word in vecs:
                    file_vecs.append(vecs[word])
        s = np.sum(x for x in file_vecs)
        s /= len(file_vecs)

        X.append(s)
        y.append(is_entertainment(cat))

X = np.array(X)

X_scaled = preprocessing.scale(X)

print(X.shape)

# neigh = KNeighborsClassifier(n_neighbors=6, algorithm='brute', metric='cosine')
neigh = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', metric='euclidean')
neigh.fit(X, y)

# print(neigh.predict([[1], [2], [3]]))

# print(neigh.predict_proba([[0.9]]))

names = []
X_test = []
true_cats = []
for true_category in categories:
    # i = 0
    print("predicting for " + true_category)
    for filename in glob.glob("data/test/" + true_category + "/*.txt"):
        file_vecs = []
        for line in open(filename, encoding="utf-8"):
            for word in line.split(" "):
                if word and word in vecs:
                    file_vecs.append(vecs[word])

        s = np.sum(x for x in file_vecs)
        s /= len(file_vecs)
        X_test.append(s)
        names.append(filename)
        true_cats.append(is_entertainment(true_category))

X_test_scaled = preprocessing.scale(X_test)
predicted = neigh.predict(X_test)
# predicted = list(map(is_entertainment, predicted))

print(predicted)
print(true_cats)

predicted = np.array(predicted)
true_cats = np.array(true_cats)

tp = ((predicted == 1) * (true_cats == 1)).sum()
fp = ((predicted == 1) * (true_cats == 0)).sum()
tn = ((predicted == 0) * (true_cats == 0)).sum()
fn = ((predicted == 0) * (true_cats == 1)).sum()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + fn + fp + tn)
F1 = 2 * (precision * recall) / (precision + recall)

acc = [[precision, recall, accuracy, F1]]
print(tabulate(acc, headers=["precision", "recall",
                             "accuracy", "F1-measure"], floatfmt=".2f"))

# for i in range(len(names)):
#     if predicted[i] == 0 and true_cats[i] == 1:
#         print(names[i])
