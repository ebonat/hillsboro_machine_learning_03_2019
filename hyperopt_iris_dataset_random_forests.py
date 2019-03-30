
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import time

# def objective_function(args):
# 
#     if args['model']==KNeighborsClassifier:
#         n_neighbors = args['param']['n_neighbors']
#         algorithm = args['param']['algorithm']
#         leaf_size = args['param']['leaf_size']
#         metric = args['param']['metric']
#         clf = KNeighborsClassifier(n_neighbors=n_neighbors,
#                                algorithm=algorithm,
#                                leaf_size=leaf_size,
#                                metric=metric,
#                                )
#     elif args['model']==SVC:
#         C = args['param']['C']
#         criterion = args['param']['criterion']
#         max_depth = args['param']['max_depth']
#         max_features = args['param']['max_features']
#         clf = RandomForestClassifier()
# 
#     clf.fit(X_train,y_train)
# 
#     y_pred_train = clf.predict(X_train)
#     loss = mean_squared_error(y_train,y_pred_train)
#     print("Test Score:",clf.score(X_test,y_test))
#     print("Train Score:",clf.score(X_train,y_train))
#     print("\n=================")
#     return loss

def objective_function(args):
    n_estimators = args['param']['n_estimators']
    criterion = args['param']['criterion']
    max_depth = args['param']['max_depth']
    max_features = args['param']['max_features']
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, 
                                 max_depth=max_depth, max_features=max_features)

    clf.fit(X_train,y_train)

    y_pred_train = clf.predict(X_train)
    loss = mean_squared_error(y_train,y_pred_train)
    print("Test Score:",clf.score(X_test,y_test))
    print("Train Score:",clf.score(X_train,y_train))
    print("\n=================")
    return loss

def main():
    global X_train, X_test,y_train,y_test

    iris = datasets.load_iris() 
    X = iris.data
    y = iris.target

    X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2)
    
    search_space = {'param':{'n_estimators': hp.choice('n_estimators', range(100, 500)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'max_depth': hp.choice('max_depth', range(1,100)),
        'max_features': hp.choice('max_features', ["auto","sqrt","log2"])}}
#         'min_samples_split': hp.choice('min_samples_split', [2, 3, 4, 5]),
#         'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 3, 4, 5]),
#         'bootstrap': hp.choice('bootstrap', [True, False]),
#         }        
    
#     hyperparameter_space2 = {'param':{'C':hp.uniform('C',0,1), 
#                                       'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']), 
#                                       'degree':hp.choice('degree',range(1,15)), 
#                                       'gamma':hp.uniform('gamma',0.001,10000)}}

#     best_classifier = fmin(objective_func, space, algo=tpe.suggest, max_evals=100)    
    best_classifier = fmin(fn=objective_function, space=search_space, algo=tpe.suggest, 
                           max_evals=100, verbose=0)
    
    print(best_classifier)
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    seconds = str(round(end_time - start_time, 1))
    minutes = str(round((end_time - start_time) / 60, 1))
    print()
    print("Program Runtime:")
    print("Seconds: {} | Minutes: {}".format(seconds, minutes))