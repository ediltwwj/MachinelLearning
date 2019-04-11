import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from KNN.modelSplit import train_test_split
from KNN.kNN import kNNClassifier
from ModelTest.metrics import accuracy_score


digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

# 还原特征变为图像
# some_digit = X[666]
# print(y[666])
# some_digit_image = some_digit.reshape(8, 8)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary)
# plt.show()

my_knn_clf = kNNClassifier(k = 3)
my_knn_clf.fit(X_train, y_train)
"""只关心准确率，不关心预测结果"""
print("自定义kNN模型预测准确率为：{}".format(my_knn_clf.score(X_test, y_test)))

"""使用自定义KNN模型测试手写数字准确度"""
y_predict = my_knn_clf.predict(X_test)
print("自定义kNN模型预测准确率为：{}".format(accuracy_score(y_test, y_predict)))

