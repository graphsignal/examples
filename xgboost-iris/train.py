import logging
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import joblib
from sklearn.metrics import precision_score
# Graphsignal: import module
import graphsignal
from graphsignal.profilers.xgboost import GraphsignalCallback

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: configure module
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(workload_name='XGBoost Iris')


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'nthread': 2,
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'multi:softprob',
    'eval_metric': ['auc'],
    'num_class': 3}
num_round = 20

# Graphsignal: add profiler callback
bst = xgb.train(
    params, 
    dtrain, 
    num_round, 
    evals=[(dtrain, 'Train'), (dtest, 'Valid')],
    callbacks=[GraphsignalCallback()])
