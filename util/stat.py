import numpy as np
from scipy import stats
from sklearn import metrics
import torch

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def compute_unweighted_accuracy(list1, list2):
    result = []
    for i in range(len(list1)):
        result.append(list1[i] / list2[i])
    return sum(result)/len(result)

def compute_weighted_f1(tp, fp, fn, unweightet_total):
    f1_scores = []
    num_classes = len(tp)
    
    for i in range(num_classes):
        if tp[i] + fp[i] == 0:
            precision = 0
        else:
            precision = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall = 0
        else:
            recall = tp[i] / (tp[i] + fn[i])
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
            
    wf1 = sum([f1_scores[i] * unweightet_total[i] for i in range(num_classes)]) / sum(unweightet_total)
    return wf1

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = output.shape[-1]
    stats = []

    output = np.argmax(output, 1)

    correct, total = 0, 0

    # unweighted accuracy
    unweightet_correct = [0] * classes_num
    unweightet_total = [0] * classes_num

    # weighted f1
    tp = [0] * classes_num
    fp = [0] * classes_num
    fn = [0] * classes_num

    for i in range(len(target)):
        total += 1
        correct += (output[i] == target[i])
        unweightet_total[target[i]] += 1
        if output[i] == target[i]:
            unweightet_correct[target[i]] += 1
            tp[target[i]] += 1
        else:
            fp[output[i]] += 1
            fn[target[i]] += 1

    weighted_acc = correct / total * 100
    unweighted_acc = compute_unweighted_accuracy(unweightet_correct, unweightet_total) * 100
    weighted_f1 = compute_weighted_f1(tp, fp, fn, unweightet_total) * 100
    
    dict = {'weighted_acc': weighted_acc,
            'unweighted_acc': unweighted_acc,
            'weighted_f1': weighted_f1
            }
    stats.append(dict)
    return stats