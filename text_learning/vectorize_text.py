#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append("../tools/")
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

from_sara = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

# temp_counter is a way to speed up the development--there are
# thousands of emails from Sara and Chris, so running over all of them
# can take a long time
# temp_counter helps you only look at the first 200 emails in the list so you
# can iterate your modifications quicker
sw = ["sara", "shackleton", "chris", "germani"]
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        # only look at first 200 emails when developing
        # once everything is working, remove this line to run over full dataset
        path = os.path.join('..', path[:-1])
        print path
        email = open(path, "r")

        text = str(parseOutText(email))
        for rm in sw:
            if(rm in text):
                text = text.replace(rm,"")

        word_data.append(text)

        if name == "Sara":
            from_data.append(0)
        if name == "Chris":
            from_data.append(1)

        email.close()

print "emails processed"
from_sara.close()
from_chris.close()
print word_data[152]

pickle.dump(word_data, open("your_word_data.pkl", "w"))
pickle.dump(from_data, open("your_email_authors.pkl", "w"))

# in Part 4, do TfIdf vectorization here
vectorizer = TfidfVectorizer(stop_words='english')
vec_fit=vectorizer.fit_transform(word_data)

print len(vectorizer.get_feature_names())
vec_words = vectorizer.get_feature_names()
print vec_words[34597]
