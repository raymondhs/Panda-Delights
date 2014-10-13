# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys

'''
projects = pd.read_csv('../data/projects.csv')
projects_arr = np.array(projects)
print "Done reading project data"
'''
essays = pd.read_csv('../data/essays.csv')
essays_arr = np.array(essays)
print "Done reading essay data"

if sys.argv[1] == "sample":
    sample_train = {}
    for pid in open('../data/sample_01.txt'):
        sample_train[pid.strip()] = True
    '''
    project_idx = np.array([x for x in xrange(projects_arr.shape[0]) if projects_arr[x][0] in sample_train])
    project_data = pd.DataFrame(projects_arr[project_idx,:], columns = list(projects))
    project_data.to_csv('../data/projects_sample_01.csv', index=False)
    print "Done writing project data"
    '''

    ess_idx = np.array([x for x in xrange(essays_arr.shape[0]) if essays_arr[x][0] in sample_train])
    ess_data = pd.DataFrame(essays_arr[ess_idx,:], columns = list(essays))
    ess_data.to_csv('../data/essays_sample_01.csv', index=False)
    print "Done writing essay data"
else:
    project_idx = \
        np.where((projects_arr[:,-1] < '2014-01-01') \
              & (projects_arr[:,-1] >= '2010-04-01'))[0]

    project_data = pd.DataFrame(projects_arr[project_idx,:], columns = list(projects))
    project_data.to_csv('../data/projects_train_postapril10.csv', index=False)
    print "Done writing project data"
