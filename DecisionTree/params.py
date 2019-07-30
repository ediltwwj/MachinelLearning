from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from LogisticRegression.drawDecisionBoundary import plot_decision_boundary

if __name__ == "__main__":

    X, y = datasets.make_moons(noise=0.25, random_state=666)

    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    # 使用默认参数
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X, y)

    plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    # 使用max_depth参数
    dt_clf2 = DecisionTreeClassifier(max_depth=2)
    dt_clf2.fit(X, y)

    plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    # 使用min_samples_split
    # 对于一个节点，至少拥有min_samples_split个样本才继续进行分割
    dt_clf3 = DecisionTreeClassifier(min_impurity_split=10)
    dt_clf3.fit(X, y)

    plot_decision_boundary(dt_clf3, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

    # min_samples_leaf 对于一个叶子节点至少有多少个样本，容易产生过拟合
    # max_leaf_nodes 最多有多少个叶子节点