import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ConfusionMatrix.func import *
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


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
    y_predict = log_reg.predict(X_test)
    decision_scores = log_reg.decision_function(X_test)

    fprs = []
    tprs = []
    thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)

    for threshold in thresholds:
        y_predict = np.array(decision_scores >= threshold, dtype='int')
        fprs.append(FPR(y_test, y_predict))
        tprs.append(TPR(y_test, y_predict))

    # 绘制ROC曲线
    plt.plot(fprs, tprs)
    plt.show()

    # 使用scikit-learn中的ROC曲线
    fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
    plt.plot(fprs, tprs)
    plt.show()

    # 曲线底下面积越大，分类算法越好
    # 可以比较模型优劣，面积大的优
    print(roc_auc_score(y_test, decision_scores))