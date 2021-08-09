
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection as sk_ms
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, f1_score


RANDOM_SEED = 20
FRAC_TRAIN = 0.8
class Classification(object):

    def __init__(self, features, labels, kfold):
        self.features = features
        self.labels = labels
        self.kfold = kfold

    def classification(self):
        c1 = DecisionTreeClassifier(random_state=0)
        c2 = KNeighborsClassifier(n_neighbors=5) ## testar outros parametros 3 41.6666666  ### 5 45.
        c3 = GaussianNB()
        c4 = SVC(kernel='linear', probability=True)
        classifiers = [c1,c2,c3,c4]
        results = []
        stds = []
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, stratify=self.labels, test_size=(1.0 - FRAC_TRAIN), random_state=RANDOM_SEED)
        for classifier in classifiers: 
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print("Score {}".format(score))
            #scores = sk_ms.cross_val_score(i, self.features, self.labels, cv=self.kfold, scoring='accuracy', n_jobs=-1, verbose=0)
            #score = round(scores.mean() * 100, 2)
            #sd = round(scores.std()*100, 2)
            results.append(score)
            stds.append(score)
        return results, stds

    def get_scores(self):
        return np.array(self.classification())
