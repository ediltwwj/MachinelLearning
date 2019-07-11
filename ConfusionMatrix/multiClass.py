import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import  confusion_matrix

if __name__ == "__main__":

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_tarin, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

    log_reg = LogisticRegression()
    log_reg.fit(X_tarin, y_train)
    y_predict = log_reg.predict(X_test)
    print("Score = %s" %log_reg.score(X_test, y_test))
    # 提示错误，Target is multiclass but average='binary'.
    # 说明默认参数只可以解决二分类问题
    # print("Precision Score  = %s" %precision_score(y_test, y_predict))
    print("Precision Score = %s" %precision_score(y_test, y_predict, average='micro'))

    # 多分类的混淆矩阵
    # 对角线代表这个数字预测正确的个数
    # 行代表真实值，列代表预测错误的值
    cfm = confusion_matrix(y_test, y_predict)
    print(cfm)

    # 绘制混淆矩阵
    plt.matshow(cfm, cmap=plt.cm.gray)
    plt.show()

    # 对混淆矩阵进行数据提取
    # 提取每行的样本数
    row_sums = np.sum(cfm, axis=1)
    err_matrix = cfm / row_sums
    # 对角线的值设置为0
    # 越亮代表预测错误越多
    np.fill_diagonal(err_matrix, 0)
    plt.matshow(err_matrix, cmap=plt.cm.gray)
    plt.show()


