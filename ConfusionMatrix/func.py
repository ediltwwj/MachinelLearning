import numpy as np

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

def f1_score(precision, recall):
    """平衡精准率和召回率"""

    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return  0.0

def TPR(y_true, y_predict):

    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        0.0

def FPR(y_true, y_predict):

    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        0.0