from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression

"""使用scikit-learn的Linear Regression算法"""

if __name__ == "__main__":
    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    print("Scikit-Learn's Linear Regression Score : {}".format(lin_reg.score(X_test, y_test)))
