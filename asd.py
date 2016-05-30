import pandas as pd
import csv
data_white_both = pd.read_csv('winequality-white.csv', sep=';')
data_red_both = pd.read_csv('winequality-red.csv', sep=';')
X_train_white, X_test_white, Y_train_white, Y_test_white = train_test_split(data_white_both[data_white_both.columns[:11]], data_white_both['quality'], test_size=0.4, train_size=0.6)
X_train_red, X_test_red, Y_train_red, Y_test_red = train_test_split(data_red_both[data_red_both.columns[:11]], data_red_both['quality'], test_size=0.4, train_size=0.6)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
C_param_list = [10**(i) for i in range(-4, 2)]
Kern_parampampam = ['rbf', 'linear', 'poly']
accu = {key:[] for key in Kern_parampampam}
clock = {key:[] for key in Kern_parampampam}
for C in C_param_list:
    for K in Kern_parampampam:
        start = time.time()
        clf = SVC(C=C, kernel=K)
        clf.fit(X_train_white, Y_train_white)
        pred = clf.predict(X_test_white)
        accu[K].append(accuracy_score(pred, Y_test_white))
        end = time.time()
        clock[K].append(end - start)
from joblib import dump
dump(accu, 'result')
