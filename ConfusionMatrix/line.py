import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve


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

    precisions = []  # 存放每一个阈值对应的精准率
    recalls = []  # 存放每一个阈值对应的召回率
    thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)  # 阈值间距为0.1

    for threshold in thresholds:
        y_predict = np.array(decision_scores >= threshold, dtype='int')
        precisions.append(precision_score(y_test, y_predict))
        recalls.append(recall_score(y_test, y_predict))

    plt.plot(thresholds, precisions)
    plt.plot(thresholds, recalls)
    plt.show()


    # Precision-Recall曲线
    plt.plot(precisions, recalls)
    plt.show()

    # 使用scikit-learn中的Precision-Recall曲线
    precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)
    plt.plot(thresholds, precisions[:-1])
    plt.plot(thresholds, recalls[:-1])
    plt.show()

    plt.plot(precisions, recalls)
    plt.show()

