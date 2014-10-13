# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys, random

projects = pd.read_csv('../data/projects.csv')
projects_arr = np.array(projects)
print "Done reading project data"

random.seed(19)

if sys.argv[1] == "sample":
    project_idx = \
        np.where((projects_arr[:,-1] < '2014-01-01') \
              & (projects_arr[:,-1] >= '2010-04-01'))[0]
    sample = random.sample(project_idx.tolist(), 6000)
    sample = np.array(sample)
    project_data = pd.DataFrame(projects_arr[sample,0])
    project_data.to_csv('../data/sample_01_projectids.txt', header=None, index=False)

if sys.argv[1] == "train":
    project_idx = \
        np.where((projects_arr[:,-1] < '2014-01-01') \
              & (projects_arr[:,-1] >= '2010-04-01'))[0]
    project_data = pd.DataFrame(projects_arr[project_idx,0])
    project_data.to_csv('../data/train_projectids.txt', header=None, index=False)

if sys.argv[1] == "test":
    project_idx = \
        np.where(projects_arr[:,-1] >= '2014-01-01')[0]
    project_data = pd.DataFrame(projects_arr[project_idx,0])
    project_data.to_csv('../data/test_projectids.txt', header=None, index=False)

print "Done writing project data"
