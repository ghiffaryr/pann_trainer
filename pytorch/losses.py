import torch
import torch.nn.functional as F
import sklearn


def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy_with_logits(
        output_dict['clipwise_output'], target_dict['target'])

def clip_balanced_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    class_weights = class_weight.compute_class_weight('balanced',np.unique(target_dict['target']),target_dict['target'].numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return F.binary_cross_entropy_with_logits(
        output_dict['clipwise_output'], target_dict['target'], weight=class_weights)

def clip_nll(output_dict, target_dict):
    loss = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])
    return loss

def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    if loss_type == 'clip_balanced_bce':
        return clip_bce
    if loss_type == 'clip_nll':
        return clip_nll