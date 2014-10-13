# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys

projects = pd.read_csv('../data/projects.csv')
projects_arr = np.array(projects)
print "Done reading project data"

essays = pd.read_csv('../data/essays.csv')
essays_arr = np.array(essays)
print "Done reading essay data"

outcomes = pd.read_csv('../data/outcomes.csv')
outcomes_arr = np.array(outcomes)
print "Done reading outcome data"

if sys.argv[1] == "sample":
    ids = '../data/sample_01_projectids.txt'
    project_out = '../data/projects_sample_01.csv'
    ess_out = '../data/essays_sample_01.csv'
    outcomes_out = '../data/outcomes_sample_01.csv'
elif sys.argv[1] == "train":
    ids = '../data/train_projectids.txt'
    project_out = '../data/projects_train.csv'
    ess_out = '../data/essays_train.csv'
    outcomes_out = '../data/outcomes_train.csv'
elif sys.argv[1] == "test":
    ids = '../data/test_projectids.txt'
    project_out = '../data/projects_test.csv'
    ess_out = '../data/essays_test.csv'

sample_train = {}
for pid in open(ids):
    sample_train[pid.strip()] = True

project_idx = np.array([x for x in xrange(projects_arr.shape[0]) if projects_arr[x][0] in sample_train])
project_data = pd.DataFrame(projects_arr[project_idx,:], columns = list(projects))
project_data.to_csv(project_out, index=False)
print "Done writing project data"

ess_idx = np.array([x for x in xrange(essays_arr.shape[0]) if essays_arr[x][0] in sample_train])
ess_data = pd.DataFrame(essays_arr[ess_idx,:], columns = list(essays))
ess_data.to_csv(ess_out, index=False)
print "Done writing essay data"

if sys.argv[1] != "test":
    outcomes_idx = np.array([x for x in xrange(outcomes_arr.shape[0]) if outcomes_arr[x][0] in sample_train])
    outcomes_data = pd.DataFrame(outcomes_arr[outcomes_idx,:], columns = list(outcomes))
    outcomes_data.to_csv(outcomes_out, index=False)
    print "Done writing outcome data"
