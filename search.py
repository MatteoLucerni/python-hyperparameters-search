import matplotlib.pyplot as mlt
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from GPyOpt.methods import BayesianOptimization
import pandas as pd
import numpy as np
import time

iris = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=["sepal length", "sepal width", "petal length", "petal width", "class"],
)
iris.head()

X = iris.drop("class", axis=1).values
Y = iris["class"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# GridSearch
svc = SVC()

params = {
    "kernel": ["linear", "sigmoid", "rbf", "poly"],
    "C": [1, 10, 100, 1000],
    "gamma": [0.1, 1, "auto"],
}

gs = GridSearchCV(svc, params, cv=10)

start = time.time()

gs.fit(X_train, Y_train)

print("=" * 100)

print(f"Time GS: {round(time.time() - start, ndigits=3)}s")

print(f"Best params: {gs.best_params_}")
print(f"Best accuracy: {gs.best_score_}")

svc = gs.best_estimator_

final_score = svc.score(X_test, Y_test)

print(f"Final score: {final_score}")

print("=" * 100)

# RandomSearch
svc = SVC()

params = {
    "kernel": ["linear", "sigmoid", "rbf", "poly"],
    "C": [1, 10, 100, 1000],
    "gamma": [0.1, 1, "auto"],
}

rs = RandomizedSearchCV(svc, params, cv=10)

start = time.time()

rs.fit(X_train, Y_train)

print(f"Time RS: {round(time.time() - start, ndigits=3)}s")

print(f"Best params: {rs.best_params_}")
print(f"Best accuracy: {rs.best_score_}")

svc = rs.best_estimator_

final_score = svc.score(X_test, Y_test)

print(f"Final score: {final_score}")

print("=" * 100)

# BayesOptimization
kernel_mapping = {0: "rbf", 1: "linear", 2: "poly", 3: "sigmoid"}

domain = [
    {"name": "kernel", "type": "discrete", "domain": (0, 3)},
    {"name": "C", "type": "continuous", "domain": (0.1, 100)},
    {"name": "gamma", "type": "continuous", "domain": (0.01, 1)},
]


def objective_function(params):
    svc = SVC(
        kernel=kernel_mapping[int(params[0, 0])], C=params[0, 1], gamma=params[0, 2]
    )
    skfold = StratifiedKFold(n_splits=10, shuffle=True)

    scores = []
    for train_index, val_index in skfold.split(X_train, Y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = Y_train[train_index], Y_train[val_index]

        svc.fit(X_train_fold, y_train_fold)
        score = svc.score(X_val_fold, y_val_fold)
        scores.append(score)

    return np.mean(scores)


bo = BayesianOptimization(
    f=objective_function,
    domain=domain,
    acquisition_type="EI",
    initial_design="latin_hypercube",
    maximize=True,
)

start = time.time()
bo.run_optimization(max_iter=50)
print(f"Time BO: {round(time.time() - start, ndigits=3)}s")

best_kernel_index = int(bo.x_opt[0])
best_C = bo.x_opt[1]
best_gamma = bo.x_opt[2]

best_kernel = kernel_mapping[best_kernel_index]

best_svc = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma)
best_svc.fit(X_train, Y_train)

print(f"Best params: kernel: '{best_kernel}', gamma: '{best_gamma}', C: '{best_C}'")

final_score = best_svc.score(X_test, Y_test)
print(f"Final score: {final_score}")


print("=" * 100)
