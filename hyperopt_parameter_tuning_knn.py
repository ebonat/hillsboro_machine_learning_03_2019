
# Parameter Tuning with Hyperopt
# https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

iris = datasets.load_iris()
X = iris.data
y = iris.target

def hyperopt_train_test(params):
    classifier = KNeighborsClassifier(**params)    
    cv_fold = KFold(n_splits=3, shuffle=True, random_state=1)
    score = cross_val_score(classifier, X, y, cv=cv_fold, scoring='roc_auc', n_jobs=-1)    
    return score.mean()

def hyperopt_train_test2(params):
    clf = KNeighborsClassifier(**params)
    score = cross_val_score(clf, X, y, cv=3, n_jobs=1)
    return score.mean()

search_space = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
    'weights': hp.choice('weights ', ["uniform", "distance"]),
    'algorithm': hp.choice('algorithm', ["auto", "ball_tree", "kd_tree", "brute"])
}

def objective_function(params):
    acc = hyperopt_train_test2(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(objective_function, search_space, algo=tpe.suggest, max_evals=100, trials=trials)
print("best: " + str(best))

