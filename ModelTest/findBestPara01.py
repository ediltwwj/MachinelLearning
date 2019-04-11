import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def getBestPara():
    """自定义寻找KNN算法之手写数字模型的最好的超参数"""

    best_p = -1
    best_score = 0.0  # 测试准确率
    best_k = -1  # 最佳超参数

    for k in range(1, 11):
        for p in range(1, 6):
            knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p) # 考虑权重(距离),p对应不同的明可夫斯基距离
            knn_clf.fit(X_train, y_train)
            score = knn_clf.score(X_test, y_test)

            if score > best_score:
                best_p = p;
                best_k = k
                best_score = score

    return best_p, best_k, best_score


if __name__ == "__main__":

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

    best_p, best_k, best_score = getBestPara()

    print("Best_pd:{}, Best_k:{},Best_score:{}".format(best_p, best_k, best_score))
