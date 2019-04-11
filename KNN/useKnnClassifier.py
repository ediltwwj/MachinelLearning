from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split


"""使用scikit-learn库的KNN算法解决分类任务"""

if __name__ == "__main__":

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    kNN_classifier = KNeighborsClassifier(n_neighbors=6)  # 创建kNN实例，传入k值
    kNN_classifier.fit(X_train, y_train)  # 拟合模型, 传入数据集和特征集

    print("kNN分类算法模型准确度 : {}".format(kNN_classifier.score(X_test, y_test)))




