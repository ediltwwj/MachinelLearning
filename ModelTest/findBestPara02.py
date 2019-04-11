import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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
    """利用kNN网格搜索数字模型最好的超参数(分类任务)"""

    knn_clf = KNeighborsClassifier()

    grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)  # n_jobs并行创建分类器，-1即使用本机全部的核， verbose输出详细信息
    grid_search.fit(X_train, y_train)

    # 运行较慢，耐心等待结果
    print("Best Estimator List : {}".format(grid_search.best_estimator_))  # 打印最好的参数
    print("Best Predict Score : {}".format(grid_search.best_score_))  # 打印最高的准确率
    print("Best Params List : {}".format(grid_search.best_params_))  # 打印自定义中最好的参数

    return grid_search.best_estimator_


if __name__ == "__main__":
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

    knn_clf = getBestPara()  # 使用最佳模型

    print("After Select Best Params, the predict score is : {}".format(knn_clf.score(X_test, y_test)))
