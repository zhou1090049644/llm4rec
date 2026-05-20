import sklearn
from sklearn.metrics import roc_auc_score


def manual_auc(y_true, y_pred):
    # 合并标签和预测值，并按预测值升序排序
    data = sorted(zip(y_pred, y_true), key=lambda x: x[0])
    pred_sorted, labels_sorted = zip(*data)
    print(pred_sorted)
    print(labels_sorted)

    # 计算正样本的秩次（从 1 开始）
    ranks = []
    for i, (pred, label) in enumerate(data):
        if label == 1:
            ranks.append(i + 1)  # 索引从 0 开始，秩次从 1 开始
    print(ranks)

    # 计算曼-惠特尼 U 统计量
    m = sum(labels_sorted)  # 正样本数
    n = len(labels_sorted) - m  # 负样本数
    sum_ranks = sum(ranks)
    auc = (sum_ranks - m * (m + 1) / 2) / (m * n)
    return auc

def auc_score(y_true, y_pred):
    pos = []
    neg = []

    for y, s in zip(y_true, y_pred):
        if y == 1:
            pos.append(s)
        else:
            neg.append(s)
        
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    
    correct = 0.0

    for p in pos:
        for n in neg:
            if  p > n:
                correct += 1
            elif p == n:
                correct += 0.5
    return correct / (len(pos) * len(neg))
    


# 测试
y_true = [0, 0, 1, 1, 0]
y_pred = [0.1, 0.4, 0.35, 0.8, 0.2]
print(f"手动计算 AUC: {manual_auc(y_true, y_pred):.4f}")
# print(f"sklearn 计算 AUC: {roc_auc_score(y_true, y_pred):.4f}")
# print(f"手动计算(原始概率法) AUC:{auc_score(y_true, y_pred):.4f}")

