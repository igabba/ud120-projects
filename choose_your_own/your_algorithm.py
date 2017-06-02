#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

features_train, labels_train, features_test, labels_test = makeTerrainData()

# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary

# clf = SVC(kernel='rbf', C=10000) #0.932
# clf = DecisionTreeClassifier(random_state=0, criterion='entropy') #0.904
max_accuracy = 0
max_n_neighbors = 1
for i in range(1, 100):
    clf = KNeighborsClassifier(n_neighbors=i)  # 0.94
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    accuracy = accuracy_score(pred, labels_test)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        max_n_neighbors = i

print max_accuracy
print max_n_neighbors
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
