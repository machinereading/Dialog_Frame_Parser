def precision_1d(gold, pred):
    gold = set(gold)
    pred = set(pred)
    n_correct = len(gold & pred)
    n_pred = len(pred)
    score = n_correct / n_pred if n_pred > 0 else 0
    return score

def recall_1d(gold, pred):
    gold = set(gold)
    pred = set(pred)
    n_correct = len(gold & pred)
    n_true = len(gold)
    score = n_correct / n_true if n_true > 0 else 0
    return score

def f1_1d(gold, pred):
    gold = set(gold)
    pred = set(pred)
    n_correct = len(gold & pred)
    n_true = len(gold)
    n_pred = len(pred)
    p = n_correct / n_pred if n_pred > 0 else 0
    r = n_correct / n_true if n_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return score