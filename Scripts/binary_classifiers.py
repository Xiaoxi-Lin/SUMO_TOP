# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:45:17 2022

@author: 73273
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

"""construct various binary classifier"""
rfc = RandomForestClassifier(n_estimators=370,
    bootstrap=True,
    oob_score=True,
    min_samples_split=7,
    min_samples_leaf =1,
    max_features='sqrt',
    n_jobs=-1)

svc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

gbc = GradientBoostingClassifier(n_estimators=500)

