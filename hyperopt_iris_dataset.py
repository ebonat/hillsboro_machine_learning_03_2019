
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin
from sklearn.metrics import mean_squared_error
import time

def objective_function(args):

    if args['model']==KNeighborsClassifier:
        n_neighbors = args['param']['n_neighbors']
        algorithm = args['param']['algorithm']
        leaf_size = args['param']['leaf_size']
        metric = args['param']['metric']
        clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                               algorithm=algorithm,
                               leaf_size=leaf_size,
                               metric=metric,
                               )
    elif args['model']==SVC:
        C = args['param']['C']
        kernel = args['param']['kernel']
        degree = args['param']['degree']
        gamma = args['param']['gamma']
        clf = SVC(C=C, kernel=kernel, degree=degree,gamma=gamma)
    

    clf.fit(X_train,y_train)

    y_pred_train = clf.predict(X_train)
    loss = mean_squared_error(y_train,y_pred_train)
    print("Test Score:",clf.score(X_test,y_test))
    print("Train Score:",clf.score(X_train,y_train))
    print("\n=================")
    return loss

def objective_function2(args):
    C = args['param']['C']
    kernel = args['param']['kernel']
    degree = args['param']['degree']
    gamma = args['param']['gamma']
    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    

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
#     X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
    
#     hyperparameter_space = hp.choice('classifier',[
#         {'model': KNeighborsClassifier, 'param':{'n_neighbors':hp.choice('n_neighbors',range(3,11)), 'algorithm':hp.choice('algorithm',['ball_tree','kd_tree']), 'leaf_size':hp.choice('leaf_size',range(1,50)), 'metric':hp.choice('metric', ["euclidean","manhattan", "chebyshev","minkowski"])}},        
#         {'model': SVC, 'param':{'C':hp.lognormal('C',0,1), 'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']), 'degree':hp.choice('degree',range(1,15)), 'gamma':hp.uniform('gamma',0.001,10000)}}
#         ])
    
#     hyperparameter_space2 = {'param':{'C':hp.lognormal('C',0,1), 'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']), 'degree':hp.choice('degree',range(1,15)), 'gamma':hp.uniform('gamma',0.001,10000)}}
#     {'degree': 1, 'gamma': 1231.8609148337193, 'kernel': 2, 'C': 22.8538632367702}

    hyperparameter_space2 = {'param':{'C':hp.uniform('C',0,1), 'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']), 'degree':hp.choice('degree',range(1,15)), 'gamma':hp.uniform('gamma',0.001,10000)}}
#     {'kernel': 0, 'degree': 6, 'C': 3.655860649875491, 'gamma': 1028.799517037706}
    
#     hyperparameter_space2 = {'C':hp.uniform('C', 0, 20), 
#                                                               'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']), 
#                                                               'degree':hp.choice('degree',range(1,15)), 
#                                                               'gamma':hp.uniform('gamma',0, 20)}

#     best_classifier = fmin(objective_func, space, algo=tpe.suggest, max_evals=100)    
    best_classifier = fmin(fn=objective_function2, space=hyperparameter_space2, algo=tpe.suggest, max_evals=100, verbose=0)
    
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