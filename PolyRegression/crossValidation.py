import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)

    print("普通调参 ： ")
    best_score, best_p, best_k = 0, 0, 0
    # 使用循环进行调参
    for k in range(2, 11):
        for p in range(1, 6):
            knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
            knn_clf.fit(X_train, y_train)
            score = knn_clf.score(X_test, y_test)
            if score > best_score:
                best_score, best_p, best_k = score, p, k

    print("Best K = ", best_k)
    print("Best P = ", best_p)
    print("Best Score = ", best_score)

    print("使用交叉验证进行调参 ： ")
    best_score, best_p, best_k = 0, 0, 0
    # 使用交叉验证进行调参
    for k in range(2, 11):
        for p in range(1, 6):
            knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
            scores = cross_val_score(knn_clf, X_train, y_train)  # 返回不同参数组合的模型的分数,默认分为3份，cv=来调整分割份数
            score = np.mean(scores)
            if score > best_score:
                best_score, best_p, best_k = score, p, k

    print("Best K = ", best_k)
    print("Best P = ", best_p)
    print("Best Score = ", best_score)

    # 使用交叉验证拿到的超参数构造模型
    best_knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=2, p=2)
    best_knn_clf.fit(X_train, y_train)
    print("After Cross Validation， The Score is : ", best_knn_clf.score(X_test, y_test))

    print("使用网格搜索配合交叉验证进行调参 : ")
    # 使用网格搜索进行调参
    param_grid = [
        {
            'weights' : ['distance'],
            'n_neighbors' : [i for i in range(2, 11)],
            'p' : [i for i in range(1, 6)]
        }
    ]

    grid_search = GridSearchCV(knn_clf, param_grid, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best Score : ", grid_search.best_score_)
    print("Best Params : ", grid_search.best_params_)

    best_knn_clf = grid_search.best_estimator_  # 返回最好的模型
    print("Best Score : ", best_knn_clf.score(X_test, y_test))
