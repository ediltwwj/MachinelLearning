from KNN.modelSplit import train_test_split
from KNN.kNN import kNNClassifier
from sklearn import datasets
from ModelTest.metrics import accuracy_score

"""自定义KNN算法测试"""

iris = datasets.load_iris()  # 导入鸢尾花数据集
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)  # 对数据集进行拆分，训练数据集，测试数据集

my_knn_clf = kNNClassifier(k=3)
my_knn_clf.fit(X_train, y_train)

y_predict = my_knn_clf.predict(X_test)  # 返回预测结果集
print("预测准确率为：{}".format(accuracy_score(y_test, y_predict)))
