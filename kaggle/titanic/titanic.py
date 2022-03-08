import os
from sklearn import svm

test = os.open('../data/test.csv', flags=os.O_RDONLY)
train = os.open('../data/train.csv', flags=os.O_RDONLY)
submission = os.open('../data/gender_submission.csv', flags=os.O_RDONLY)

svm.SVC()
