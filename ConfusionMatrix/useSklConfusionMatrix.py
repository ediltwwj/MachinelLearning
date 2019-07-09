import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


if __name__ == "__main__":

    # 手动形成极度偏差数据
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target.copy()

    y[digits.target==9] = 1
    y[digits.target!=9] = 0

    X_tarin, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    log_reg = LogisticRegression()
    log_reg.fit(X_tarin, y_train)
    y_log_predict = log_reg.predict(X_test)

    print("混淆矩阵 : {}".format(confusion_matrix(y_test, y_log_predict)))
    print("准准率 : {}".format(precision_score(y_test, y_log_predict)))
    print("召回率 : {}".format(recall_score(y_test, y_log_predict)))