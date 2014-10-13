# -*- coding: utf-8 -*-

"""

Beating the benchmark @ KDD 2014

__author__ : Abhishek Thakur

"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
        except:
            return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()


projects = pd.read_csv('../data/projects_sample+test.csv')
outcomes = pd.read_csv('../data/outcomes_sample_01.csv')
sample = pd.read_csv('../data/sampleSubmission.csv')
essays = pd.read_csv('../data/essays_sample+test.csv')


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
ess_proj['essay'] = ess_proj['essay'].apply(clean)
ess_proj_arr = np.array(ess_proj)
print "convert successfully"

train_idx = np.where(ess_proj_arr[:,-1] < '2014-01-01')[0]
test_idx = np.where(ess_proj_arr[:,-1] >= '2014-01-01')[0]

traindata = ess_proj_arr[train_idx,:]
testdata = ess_proj_arr[test_idx,:]
del ess_proj_arr
tfidf = TfidfVectorizer(min_df=3,  max_features=1000)

print "Training start"

tfidf.fit(traindata[:,5])
tr = tfidf.transform(traindata[:,5])
del traindata
ts = tfidf.transform(testdata[:,5])
del testdata
print "Transform finished"


lr = linear_model.LogisticRegression()
lr.fit(tr, labels=='t')
preds =lr.predict_proba(ts)[:,1]

print "Learning finished"

sample['is_exciting'] = preds
sample.to_csv('predictions.csv', index = False)
