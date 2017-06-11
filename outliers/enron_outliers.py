#!/usr/bin/python

import pickle
import sys

import matplotlib.pyplot

sys.path.append("../tools/")
from feature_format import featureFormat


def getKey(item):
    return item[0]


### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]
data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features)
data = sorted(data, key=getKey)
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    if bonus == 0:
        person = {k: v for k, v in data_dict.items() if v['salary'] == salary and (v['bonus'] == 'NaN')}
    else:
        person = {k: v for k, v in data_dict.items() if v['salary'] == salary and (v['bonus'] == bonus)}
    print person.keys().pop(0) + ' salary: ' + str(salary) + ' bonus: ' + str(bonus)
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
