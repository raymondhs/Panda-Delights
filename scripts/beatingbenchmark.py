# -*- coding: utf-8 -*-

"""

Beating the benchmark @ KDD 2014

__author__ : Abhishek Thakur
__revisions_: Raymond, Muthu, Peter

"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation

import re, sys

def clean(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
        except:
            return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()

def print_cm(cm, labels):
    """
    pretty print for confusion matrixes
    https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels])
    # Print header
    print " " * columnwidth,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "%{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            print "%{0}d".format(columnwidth) % cm[i, j],
        print

def run_cv(X, y):
    print "Running CV"
    kf = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True, random_state=1)
    it = 1
    for train_index, test_index in kf:
        print "* Iteration %d" % it
        it += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        lr = linear_model.LogisticRegression()
        lr.fit(X_train, y_train)
        preds = lr.predict_proba(X_test)[:,1]
        preds_bin = lr.predict(X_test)	
    
        cm = confusion_matrix(y_test,preds_bin,[1,0])
        print_cm(cm,["t","f"])

        '''
        f1 = f1_score(y_test, preds_bin, average=None)
        print "F1 =", f1

        acc = accuracy_score(y_test,preds_bin)
        print "ACC =", acc
        '''

        auc = roc_auc_score(y_test, preds)
        print "AUC =", auc
    

def bool2int(arr):
    newArr = np.copy(arr)
    newArr[arr=="t"] = 1
    newArr[arr=="f"] = 0
    return newArr.astype(np.int)

projects = pd.read_csv('../data/sample+test/projects.csv')
outcomes = pd.read_csv('../data/sample/outcomes.csv')
sample = pd.read_csv('../data/original/sampleSubmission.csv')
sample_bin = sample[:]
essays = pd.read_csv('../data/sample+test/essays.csv')

essays = essays.sort('projectid')
projects = projects.sort('projectid')
sample = sample.sort('projectid')
ess_proj = pd.merge(essays, projects, on='projectid')
del projects
del essays
print "finish reading"
outcomes = outcomes.sort('projectid')
outcomes_arr = np.array(outcomes)

labels = outcomes_arr[:,1]
labels = bool2int(labels)

ess_proj['essay'] = ess_proj['essay'].apply(clean)
ess_proj_arr = np.array(ess_proj)
print "convert successfully"

train_idx = np.where(ess_proj_arr[:,-1] < '2014-01-01')[0]
test_idx = np.where(ess_proj_arr[:,-1] >= '2014-01-01')[0]

traindata = ess_proj_arr[train_idx,:]
testdata = ess_proj_arr[test_idx,:]
del ess_proj_arr

'''
tfidf = TfidfVectorizer(min_df=3,  max_features=1000)

print "Training start"

tfidf.fit(traindata[:,5])
tr = tfidf.transform(traindata[:,5])
del traindata
ts = tfidf.transform(testdata[:,5])
del testdata
print "Transform finished"


'''

mode = "submit"
if len(sys.argv) > 1:
    mode = sys.argv[1]

print "running experiments"

columns = [12,13,14,15,16,17,19,20,32,33]
trt = []
tst = []
for i in xrange(10):
    subs = np.array(columns[i:])+5
    print "====================="
    print "Using %d features" % subs.shape[0]
    tr = traindata[:,subs]
    ts = testdata[:,subs]
    tr = bool2int(tr)
    ts = bool2int(ts)

    if mode == "cv":
        run_cv(tr, labels)
        continue

    lr = linear_model.LogisticRegression()
    lr.fit(tr, labels)
    preds = lr.predict_proba(ts)[:,1]
    preds_bin = lr.predict(ts)	

    print "Learning finished"
    sample['is_exciting'] = preds
    sample_bin['is_exciting'] = preds_bin

    sample.to_csv('../output/predictions_%d.csv' % i, index = False)
    sample_bin.to_csv('../output/binary_predictions/predictions_bin_%d.csv' % i, index = False)

print "====================="
print "done experiments"
   

