import glob
import operator
import re

import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import spatial
from tabulate import tabulate


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def cos(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def distance(u, v):
    return spatial.distance.euclidean(u, v)


def form_tf_idf(counts):
    transformer = TfidfTransformer(smooth_idf=False)
    return transformer.fit_transform(counts)


def is_entertainment(category):
    entertainment = ['adult', 'arts', 'games', 'home', 'society', 'shopping']
    return 1 if category in entertainment else 0


print("started")

vecs = {}

for line in open("model.txt", encoding="utf-8"):
    splitted_line = line.strip("\n").split(" ")
    word = splitted_line[0]
    vec = np.array([float(x) for x in splitted_line[1:]])
    vecs[word] = vec

print("model loaded")

data = {}
categories = []
words = []

for cats in glob.glob("data/train/*"):
    category = cats.split("\\")[-1]
    categories.append(category)
    data[category] = {}
    for filename in glob.glob("data/train/" + category + "/*.txt"):
        for line in open(filename, encoding="utf-8"):
            for word in line.split(" "):
                if not word or word not in vecs:
                    continue
                if word not in words:
                    words.append(word)

                if word in data[category]:
                    data[category][word] += 1
                else:
                    data[category][word] = 1

print("words counted")

counts = []

for category in categories:
    counts.append([])
    for word in words:
        if word in data[category]:
            counts[-1].append(data[category][word])
        else:
            counts[-1].append(0)

weights = form_tf_idf(counts)
weights = weights.toarray()

#
# weights_dict = {}
# for category in categories:
#     weights_dict[category] = {}
#     for word in words:


print("tf idf done")

cat_vecs = {}

for cat_i in range(len(categories)):
    ss = np.zeros(len(vecs[words[0]]))
    for word_i in range(len(words)):
        word = words[word_i]
        ss += np.dot(weights[cat_i][word_i], vecs[word])
    cat_vecs[categories[cat_i]] = unit_vector(ss)

print("clustered")


def predict(cat_vecs, file_vecs):
    cat_weights = {}

    s = np.sum(x for x in file_vecs)
    s = unit_vector(s)

    for category in categories:
        a = distance(cat_vecs[category], s)
        cat_weights[category] = a

    key, _ = min(cat_weights.items(), key=lambda x: x[1])
    return key


# for cat1 in cat_vecs:
#     for cat2 in cat_vecs:
#         print(1 - spatial.distance.cosine(cat_vecs[cat1], cat_vecs[cat2]), end=" ")
#     print()

predicted = []
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

        predicted.append(is_entertainment(predict(cat_vecs, file_vecs)))
        true_cats.append(is_entertainment(true_category))
        # i += 1
        # if i > 50:
        #     break
print(predicted)
print(true_cats)

predicted = np.array(predicted)
true_cats = np.array(true_cats)

tp = ((predicted == 1) * (true_cats == 1)).sum()
fp = ((predicted == 1) * (true_cats == 0)).sum()
tn = ((predicted == 0) * (true_cats == 0)).sum()
fn = ((predicted == 0) * (true_cats == 1)).sum()

# fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
# auc += metrics.auc(fpr, tpr)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + fn + fp + tn)
F1 = 2 * (precision * recall) / (precision + recall)

acc = [[precision, recall, accuracy, F1]]
print(tabulate(acc, headers=["precision", "recall",
                             "accuracy", "F1-measure"], floatfmt=".2f"))
