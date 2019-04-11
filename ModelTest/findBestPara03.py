from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]


def getBestPara():
    """利用kNN网格搜索数字模型最好的超参数(回归任务)"""

    knn_reg = KNeighborsRegressor()

    grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best Para : {}".format(grid_search.best_params_))

    return grid_search.best_estimator_


if __name__ == "__main__":
    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    lin_reg = getBestPara()

    print("After Select Best Params, the predict score is : {}".format(lin_reg.score(X_test, y_test)))
