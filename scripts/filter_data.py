# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys

projects = pd.read_csv('../data/original/projects.csv')
projects_arr = np.array(projects)
print "Done reading project data"

essays = pd.read_csv('../data/original/essays.csv')
essays_arr = np.array(essays)
print "Done reading essay data"

outcomes = pd.read_csv('../data/original/outcomes.csv')
outcomes_arr = np.array(outcomes)
print "Done reading outcome data"

if sys.argv[1] == "sample":
    dir = "../data/sample/"
elif sys.argv[1] == "train":
    dir = "../data/filtered-train/"
elif sys.argv[1] == "test":
    dir = "../data/test/"

ids = dir+'projectids.txt'
project_out = dir+'projects.csv'
ess_out = dir+'essays.csv'
outcomes_out = dir+'outcomes.csv'

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
