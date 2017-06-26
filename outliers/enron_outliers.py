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
features = ["salary", "bonus", "poi"]
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("FREVERT MARK A", 0)

data = featureFormat(data_dict, features)
data = sorted(data, key=getKey)
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    poi = point[2]
    if poi != 1:
        print point[0]
        # print {k: v for k, v in data_dict.items() if v['bonus'] == bonus}

    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
