from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

"""使用scikit-learn库的KNN算法解决回归任务"""

if __name__ == "__main__":
    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    knn_reg = KNeighborsRegressor()
    knn_reg.fit(X_train, y_train)
    print("kNN回归算法模型准确度 : {}".format(knn_reg.score(X_test, y_test)))
