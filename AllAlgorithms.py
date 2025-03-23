from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
import pandas as pd

knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=1000)
svc = SVC()
DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier()
GBC = GradientBoostingClassifier()
ADA = AdaBoostClassifier()
LDA = LinearDiscriminantAnalysis()
SDG = SGDClassifier()

models = [knn, logreg, svc, DTC, RFC, GBC, ADA, LDA, SDG]

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)

for model in models:
    model.fit(X_train, y_train)
    print(f"{model} test set score",model.score(X_test, y_test))




