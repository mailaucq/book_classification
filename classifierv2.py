from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc, accuracy_score, f1_score
import numpy as np


def getProbs(model, x_test):
  if hasattr(model, "decision_function"):
    Z = model.decision_function(x_test)
  else:
    Z = model.predict_proba(x_test)
  return Z
  

def getCurves(model, x_test, y_test, n_classes, type_m="PR"):
  y_scores = getProbs(model, x_test)
  if type_m=="ROC":
    fpr = dict()
    tpr = dict()	
    for i in range(n_classes):
      fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_scores[:, i])
      metric_value = roc_auc_score(y_test[:, i], y_scores[:, i]).mean() #auc(fpr, tpr)
    aux_1, aux_2 = np.mean(list(fpr.values())), np.mean(list(tpr.values()))
  else: #if type_m=="PR":
    precision = dict()
    recall = dict()	
    for i in range(n_classes):
      precision[i], recall[i], thresholds = precision_recall_curve(y_test[:, i], y_scores[:, i])
      metric_value = average_precision_score(y_test[:, i], y_scores[:, i]).mean() #auc(recall, precision)
    aux_1, aux_2 = np.mean(list(precision.values())), np.mean(list(recall.values()))
  return [metric_value, aux_1, aux_2]


def getClassMetrics(model, x_test, y_test, n_classes, average="micro", type_m="PR"):

  [auc, precision, recall] = getCurves(model, x_test, y_test, n_classes, type_m)
  
  y_pred = model.predict(x_test)
  
  f1 = f1_score(y_test, y_pred, average=average)
  acc = accuracy_score(y_test, y_pred)
  
  return [auc, precision, recall], f1, acc
  

def getClassifier(m, c=1, g=1):
  if m == "Bayes":
    svr = GaussianNB()
  elif m == "KNN":
    svr = KNeighborsClassifier(n_neighbors=5)
  elif m == "Tree":
    svr = DecisionTreeClassifier()
  elif m == "MLP":
    svr = MLPClassifier(alpha=c)
  elif m == "SVC":
    svr = SVC(kernel='linear', C=c)
  else:
    svr = LinearSVC(C=c, loss="hinge")
  
  return svr
