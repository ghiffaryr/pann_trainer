import numpy as np
import logging
from sklearn import metrics

from pytorch_utils import forward
from utilities import get_filename
from config import idx_to_lb

def probability_to_binary(y_pred):
    max_index = np.argmax(y_pred, axis=1)

    # Create a new array of zeros with the same shape as y_pred
    binary_y_pred = np.zeros_like(y_pred)

    # Set the maximum value in each row to 1
    # We use fancy indexing to set the values to 1
    binary_y_pred[np.arange(len(y_pred)), max_index] = 1
    return binary_y_pred

class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        statistics = {}

        cm = metrics.multilabel_confusion_matrix(target, probability_to_binary(clipwise_output))
        accuracy = np.array([])
        balanced_accuracy = np.array([])
        precision = np.array([])
        recall = np.array([])
        f1 = np.array([])
        jaccard = np.array([])
        for idx, current_cm in enumerate(cm):
            tn, fp, fn, tp = current_cm.ravel()
            try:
                current_accuracy = (tp + tn) / (tn + fp + fn + tp)
            except:
                current_accuracy = 0
            try:
                current_balanced_accuracy = ((tp / (tp + fn)) + ( tn / (tn + fp))) / 2
            except:
                current_balanced_accuracy = 0
            try:
                current_precision = tp / (tp + fp)
            except:
                current_precision = 0
            try:
                current_recall = tp / (tp + fn)
            except:
                current_recall = 0
            try:
                current_f1 = (1 + (1 ** 2) ) * (tp / (tp + fp)) * (tp / (tp + fn)) / (((1 ** 2) * (tp / (tp + fp))) + (tp / (tp + fn)))
            except:
                current_f1 = 0
            try:
                current_jaccard = tp / (tp + fp + fn)
            except:
                current_jaccard = 0
                
            statistics.update({
                'cm_'+idx_to_lb[idx]: current_cm,
                'tn_'+idx_to_lb[idx]: tn,
                'fp_'+idx_to_lb[idx]: fp,
                'fn_'+idx_to_lb[idx]: fn,
                'tp_'+idx_to_lb[idx]: tp,
                'accuracy_'+idx_to_lb[idx]: current_accuracy,
                'balanced_accuracy_'+idx_to_lb[idx]: current_balanced_accuracy,
                'precision_'+idx_to_lb[idx]: current_precision,
                'recall_'+idx_to_lb[idx]: current_recall,
                'f1_'+idx_to_lb[idx]: current_f1,
                'jaccard_'+idx_to_lb[idx]: current_jaccard,
            })
            accuracy = np.append(accuracy,current_accuracy)
            balanced_accuracy = np.append(balanced_accuracy,current_balanced_accuracy)
            precision = np.append(precision,current_precision)
            recall = np.append(recall,current_recall)
            f1 = np.append(f1,current_f1)
            jaccard = np.append(jaccard,current_jaccard)

        statistics.update({
            'accuracy': np.mean(accuracy),
            'balanced_accuracy': np.mean(balanced_accuracy),
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1': np.mean(f1),
            'jaccard': np.mean(jaccard),
        })

        return statistics