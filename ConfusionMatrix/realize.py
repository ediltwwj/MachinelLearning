import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def TN(y_true, y_predict):

    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

def FP(y_true, y_predict):

    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

def FN(y_true, y_predict):

    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

def TP(y_true, y_predict):

    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

def confusion_matrix(y_true, y_predict):
    """计算混淆矩阵"""

    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])

def precision_score(y_true, y_predict):
    """求精准率"""

    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)

    try:
        return tp / (tp + fp)
    except:
        0.0

def recall_score(y_true, y_predict):
    """求召回率"""

    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)

    try:
        return tp / (tp + fn)
    except:
        0.0


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
    print("log_reg's score : {}".format(log_reg.score(X_test, y_test)))

    y_log_predict = log_reg.predict(X_test)

    print("TN = {}".format(TN(y_test, y_log_predict)))
    print("FP = {}".format(FP(y_test, y_log_predict)))
    print("FN = {}".format(FN(y_test, y_log_predict)))
    print("TP = {}".format(TP(y_test, y_log_predict)))

    print("precision's score : {}".format(precision_score(y_test, y_log_predict)))
    print("recall's score : {}".format(recall_score(y_test, y_log_predict)))



