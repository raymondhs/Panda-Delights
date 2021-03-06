"""
Dependencies: Python 2.7 or higher, numpy, scikit-learn
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys, re

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

#Helper functions
def diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]

def run_cv(X, y):
    print "Running CV"
    kf = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True, random_state=1)
    it = 1
    mean_auc = 0.0
    for train_index, test_index in kf:
        print "* Iteration %d" % it
        it += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        preds = lr.predict_proba(X_test)[:,1]
        preds_bin = lr.predict(X_test)	

        auc = roc_auc_score(y_test, preds)
        print "AUC =", auc
        mean_auc += auc
    mean_auc /= 3
    print "AVG AUC =", mean_auc

def run_cv_essay(X, y, X_essay, X_pos=None):
    print "Running CV"
    kf = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True, random_state=1)
    it = 1
    mean_auc = 0.0
    for train_index, test_index in kf:
        print "* Iteration %d" % it
        it += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_essay_train, X_essay_test = X_essay[train_index], X_essay[test_index]
       
        clf = LogisticRegression()

        clf.fit(X_essay_train, y_train)
        print "Essay data shape =", X_essay_train.shape
        ess_preds = clf.predict_proba(X_essay_test)[:,1]
        
        lr = LogisticRegression()
        lr.fit(X_train, y_train)

        preds = lr.predict_proba(X_test)[:,1]
        preds_bin = lr.predict(X_test)	

        preds = (ess_preds+preds)/2.0

        auc = roc_auc_score(y_test, preds)
        print "AUC =", auc
        mean_auc += auc
    mean_auc /= 3
    print "AVG AUC =", mean_auc
#Loading CSV files
#donations = pd.read_csv('Data/donations.csv')
projects = pd.read_csv('Data/projects.csv')
outcomes = pd.read_csv('Data/outcomes.csv')
#resources = pd.read_csv('Data/resources.csv')
sample = pd.read_csv('Data/sampleSubmission.csv')
essays = pd.read_csv('Data/essays.csv')
tagged = pd.read_csv('Data/essays_tagged.csv')

print 'Read data files.'

#Sort data according the project ID
essays = essays.sort('projectid')
projects = projects.sort('projectid')
sample = sample.sort('projectid')
outcomes = outcomes.sort('projectid')
#donations = donations.sort('projectid')
#resources = resources.sort('projectid')
tagged = tagged.sort('projectid')

#Setting training data and test data indices
dates = np.array(projects.date_posted)
#train_idx = np.where(dates < '2014-01-01')[0]
train_idx = np.where((dates < '2014-01-01') & (dates >= '2012-07-01'))[0]
test_idx = np.where(dates >= '2014-01-01')[0]

#Filling missing values
projects = projects.fillna(value={'students_reached':32})
projects = projects.fillna(method='pad') #'pad' filling is a naive way. We have better methods.
essays = essays.fillna(value="N/A")
tagged = tagged.fillna(value="N/A")

#Set target labels
pid_dict = dict((pid,True) for pid in np.array(projects)[train_idx,:][:,0])
outcomes_arr = np.array(outcomes)
labels_idx = np.array([(pid in pid_dict) for pid in outcomes_arr[:,0]])
labels = outcomes_arr[labels_idx][:,1]
#labels = np.array(outcomes.is_exciting)

#Preprocessing
projects_numeric_columns = ['students_reached',
                            'fulfillment_labor_materials',
                            'total_price_excluding_optional_support',
                            'total_price_including_optional_support']
#projects_numeric_columns += ['school_latitude']
#projects_numeric_columns += ['school_longitude']
projects_numeric_values = np.array(projects[projects_numeric_columns])

exclude_columns = []
exclude_columns += ['school_city','school_zip','school_metro','school_district','school_county','teacher_prefix']
#exclude_columns += ['school_state']
#exclude_columns += ['primary_focus_subject','primary_focus_area','secondary_focus_subject','secondary_focus_area']
#exclude_columns += ['resource_type','grade_level']

bool_columns = ['school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'eligible_double_your_impact_match', 'eligible_almost_home_match']

projects_id_columns = ['projectid', 'school_ncesid']
#projects_id_columns += ['teacher_acctid']
projects_id_columns += ['schoolid']
projects_categorial_columns = diff(diff(diff(list(projects.columns), projects_id_columns), projects_numeric_columns), 
                                   ['date_posted'])
projects_categorial_columns = diff(projects_categorial_columns, bool_columns)
projects_categorial_columns = diff(projects_categorial_columns, exclude_columns)

#projects_categorial_values = np.array(projects[projects_categorial_columns])

#projects_numeric_columns = diff(projects_numeric_columns, ['school_latitude', 'school_longitude'])


print "%d numeric:" % len(projects_numeric_columns), 
print projects_numeric_columns
print "%d categorical:" % len(projects_categorial_columns),
print projects_categorial_columns
print "%d bool:" % len(bool_columns),
print bool_columns

selected_columns = projects_numeric_columns+projects_categorial_columns+bool_columns
#selected_columns = projects_categorial_columns+bool_columns
selected_values = np.array(projects[selected_columns])

if selected_columns[0] in (projects_categorial_columns+bool_columns):
    label_encoder = LabelEncoder()
    projects_data = label_encoder.fit_transform(selected_values[:,0])
else:
    projects_data = selected_values[:,0]

for i in range(1,selected_values.shape[1]):
    if selected_columns[i] in (projects_categorial_columns+bool_columns):
        label_encoder = LabelEncoder()
        projects_data = np.column_stack((projects_data, label_encoder.fit_transform(selected_values[:,i])))
    else:
        projects_data = np.column_stack((projects_data, selected_values[:,i]))

'''
label_encoder = LabelEncoder()
projects_data = label_encoder.fit_transform(selected_values[:,0])

for i in range(1, projects_categorial_values.shape[1]):
    label_encoder = LabelEncoder()
    projects_data = np.column_stack((projects_data, label_encoder.fit_transform(projects_categorial_values[:,i])))
'''

########## Textual features ############
#Compute essay feature
essays_arr = np.array(essays)
essays_train = essays_arr[train_idx,:]

dictio = {}
tokens = 0
for i in range(essays_train.shape[0]):
    tokens_in_ess = re.findall(r'\b\w\w+\b',essays_train[i,5])
    for token in tokens_in_ess:
        dictio[token]=1
    tokens += len(tokens_in_ess)
print "Total #tokens", tokens
print "Total #token types", len(dictio.keys())

essays_test = essays_arr[test_idx,:]
tfidf = TfidfVectorizer(min_df=3)
tfidf.fit(essays_train[:,5])
essays_train_t = tfidf.transform(essays_train[:,5])
del essays_train
essays_test_t = tfidf.transform(essays_test[:,5])
del essays_test
print "Essay data shape =", essays_train_t.shape

#Compute POS tag feature
tagged_arr = np.array(tagged)
tagged_train = tagged_arr[train_idx,:]
tagged_test = tagged_arr[test_idx,:]
tfidf = TfidfVectorizer(min_df=1)
tfidf.fit(tagged_train[:,1])
tagged_train_t = tfidf.transform(tagged_train[:,1])
del tagged_train
tagged_test_t = tfidf.transform(tagged_test[:,1])
del tagged_test
print "POS data shape =", tagged_train_t.shape

essays_pos_train_t = hstack([essays_train_t, tagged_train_t]) 
essays_pos_test_t = hstack([essays_test_t, tagged_test_t]) 

clf = LogisticRegression()

clf.fit(essays_pos_train_t, labels=='t')
ess_preds = clf.predict_proba(essays_pos_test_t)[:,1]

########################################

projects_data = projects_data.astype(float)

#One hot encoding!
categorial_idx = [(f in projects_categorial_columns) for f in selected_columns] 
enc = OneHotEncoder(categorical_features=categorial_idx,sparse=True)
enc.fit(projects_data)
projects_data = enc.transform(projects_data)

print "Data shape =", projects_data.shape

#Predicting
projects_data = projects_data.tocsr()
train = projects_data[train_idx,:]
test = projects_data[test_idx,:]
print "Test data shape =", test.shape

true_cnt = (labels=='t').sum()
false_cnt = (labels=='f').sum()
print "Training =  %d +ve, %d -ve" % (true_cnt, false_cnt)

if "cv" in sys.argv:
    #run_cv(train,labels=='t')
    #run_cv_essay(train,labels=='t',essays_train_t)
    run_cv_essay(train,labels=='t',essays_pos_train_t.tocsr())
    sys.exit(0)

project_dates = np.array(projects['date_posted'])[test_idx]

date_format = '%Y-%m-%d'
discount = []
for i in range(test.shape[0]) :
	delta = datetime.strptime(project_dates[i],date_format) - datetime(2014,02,10)
	if delta.days>0 : 
		delta = datetime(2014,05,15) - datetime.strptime(project_dates[i],date_format)
		discount.append([(delta.days/94.0*0.9)+0.1])
	else : 
		discount.append([1.0])

discount = np.array(discount)

clf = LogisticRegression()

clf.fit(train, labels=='t')
preds = clf.predict_proba(test)[:,1]

# average
preds = (ess_preds+preds)/2.0
sample['is_exciting'] = preds
sample.to_csv('predictions_nodiscount.csv', index = False)

preds = preds*discount[:,0]

#Save prediction into a file
sample['is_exciting'] = preds
sample.to_csv('predictions.csv', index = False)
