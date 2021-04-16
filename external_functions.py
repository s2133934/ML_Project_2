
# Data libraries
import pandas as pd
import numpy as np
import geopy.distance as gpy
import plotly.express as pxs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pycountry as pyc
import ccy

#Web Scraping Requirement
import datapackage

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting defaults
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['figure.dpi'] = 80

# sklearn modules
import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

import urllib
import base64
from io import BytesIO, StringIO


def logistic_pipeline(X_train, y_train):
    '''Run the logistic pipeline as in the original notebook '''
    # we used one-hot encoding for the `number` feature.
    #m = LogisticRegression(penalty = 'none', fit_intercept = False, solver='lbfgs', max_iter=1000).fit(X, y)
    m = make_pipeline(
        # StandardScaler(),
        LogisticRegression())

    param_grid = {'logisticregression__C':  np.linspace(0.1, 10, 10)}

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    logreg_model = GridSearchCV(m, param_grid, cv= kf).fit(X_train, y_train)
    
    return logreg_model

def dectree_pipeline(X_train, y_train):
    '''Run the decison tree pipeline as in original notebook '''
    pipe_dectree = make_pipeline(
                DecisionTreeClassifier(
                random_state=42,
                criterion='gini'))
    parameters = {'decisiontreeclassifier__max_depth': list(range(1,20))}

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    dectree_model = GridSearchCV(pipe_dectree, param_grid = parameters, scoring="accuracy", cv=kf,return_train_score=True).fit(X_train, y_train)
    # models_tree

    y_hat_dectree = dectree_model.predict(X_test)

    return dectree_model