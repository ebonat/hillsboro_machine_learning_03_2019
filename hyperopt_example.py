
# On Using Hyperopt: Advanced Machine Learning
# https://blog.goodaudience.com/on-using-hyperopt-advanced-machine-learning-a2dde2ccece7

# NeuPy - Neural Networks in Python
# http://neupy.com/pages/home.html

# hyperopt-sklearn
# https://github.com/hyperopt/hyperopt-sklearn

# pip install hyperopt
# Successfully installed hyperopt-0.1.1 pymongo-3.7.2
# pip install networkx==1.11
# Successfully uninstalled networkx-2.1
# Successfully installed networkx-1.11

import hyperopt
from hyperopt import hp, tpe, fmin 
# import os
# import sys
import time

# creating the objective function
def objective_function(args):
    """[summary]
    
    Arguments:
        args {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    x, y = args
    f = x**2 - y**2
    return f  

def main():
#     defining the search space, we'll explore this more later
    hyperparameter_space = [hp.uniform('x',-1,1),hp.uniform('y',-2,3)]    
#     calling the hyperopt function
    best_x_y = fmin(fn=objective_function, space=hyperparameter_space, algo=tpe.suggest, max_evals=10, verbose=1)
    print(best_x_y) 

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    seconds = str(round(end_time - start_time, 1))
    minutes = str(round((end_time - start_time) / 60, 1))
    print()
    print("Program Runtime:")
    print("Seconds: {} | Minutes: {}".format(seconds, minutes))