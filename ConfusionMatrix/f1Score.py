import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score

def f1_score(precision, recall):
    """平衡精准率和召回率"""

    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return  0.0

if __name__ == "__main__":

    precision = 0.5
    recall = 0.5
    print(f1_score(precision, recall))  # 0.5

    precision = 0.1
    recall = 0.9
    print(f1_score(precision, recall))  # 0.18

    precision = 0.0
    recall = 1.0
    print(f1_score(precision, recall))  # 0.0


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

    # print("混淆矩阵 : {}".format(confusion_matrix(y_test, y_predict)))
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)

    print(f1_score(precision, recall))


    # 使用skleran的f1_score，注意参数不同
    # print(f1_score(y_test, y_predict))