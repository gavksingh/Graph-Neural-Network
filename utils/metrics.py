import numpy as np
from munkres import Munkres
from sklearn import metrics

def custom_cluster_accuracy(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    true_labels = list(set(y_true))
    num_true_classes = len(true_labels)

    pred_labels = list(set(y_pred))
    num_pred_classes = len(pred_labels)

    ind = 0
    if num_true_classes != num_pred_classes:
        for i in true_labels:
            if i in pred_labels:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    pred_labels = list(set(y_pred))
    num_pred_classes = len(pred_labels)

    if num_true_classes != num_pred_classes:
        print("Error: Number of unique classes do not match.")
        return

    cost_matrix = np.zeros((num_true_classes, num_pred_classes), dtype=int)
    for i, true_class in enumerate(true_labels):
        true_class_indices = [i1 for i1, e1 in enumerate(y_true) if e1 == true_class]
        for j, pred_class in enumerate(pred_labels):
            pred_class_indices = [i1 for i1 in true_class_indices if y_pred[i1] == pred_class]
            cost_matrix[i][j] = len(pred_class_indices)

    m = Munkres()
    cost_matrix = cost_matrix.__neg__().tolist()
    indexes = m.compute(cost_matrix)

    new_predicted_labels = np.zeros(len(y_pred))
    for i, true_class in enumerate(true_labels):
        pred_class_index = pred_labels[indexes[i][1]]
        true_class_indices = [ind for ind, elm in enumerate(y_pred) if elm == pred_class_index]
        new_predicted_labels[true_class_indices] = true_class

    accuracy = metrics.accuracy_score(y_true, new_predicted_labels)
    f1_macro = metrics.f1_score(y_true, new_predicted_labels, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predicted_labels, average="micro")
    return accuracy, f1_macro, f1_micro
