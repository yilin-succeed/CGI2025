import logging
import math
import os
import time

import numpy as np
import torch

# def get_TP_FP_FN(outputs, targets):
#     num_classes = outputs.shape[1]
#     TP = np.zeros(num_classes)
#     FP = np.zeros(num_classes)
#     FN = np.zeros(num_classes)

#     preds = np.argmax(outputs, axis=1)
#     for i in range(num_classes):
#         idxs = np.where(targets==i)[0]
#         preds_i = preds[idxs]

#         TP[i] = np.sum(preds_i==i)
#         FP[i] = np.sum(preds==i) - TP[i]
#         FN[i] = len(idxs) - TP[i]

#     return TP, FP, FN

def get_TP_FP_FN(outputs, targets):
    num_classes = outputs.shape[1]
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    preds = torch.argmax(outputs, axis=1)
    for i in range(num_classes):
        idxs = torch.where(targets==i)[0]
        preds_i = preds[idxs]

        TP[i] = torch.sum(preds_i==i)
        FP[i] = torch.sum(preds==i) - TP[i]
        FN[i] = len(idxs) - TP[i]

    return TP, FP, FN

# def get_metrics(TP, FP, FN):
#     acc = TP.sum() / (TP + FN).sum()
#     Prec = TP / (TP+ FP)
#     Recall = TP / (TP + FN)
#     # F1 = TP / (TP + 1 / 2 * (FP + FN))
#     F1 = 2 / (1 / Prec + 1 / Recall)

#     # return acc, Prec.mean(), Recall.mean(), F1.mean()
#     return acc, Prec, Recall, F1

# def get_metrics(TP, FP, FN):
#     num_classes = TP.shape[0]
#     Prec = np.zeros(num_classes)
#     Recall = np.zeros(num_classes)
#     F1 = np.zeros(num_classes)

#     cnt_Prec = 0
#     cnt_Recall = 0
#     cnt_F1 = 0


#     if (TP + FN).sum() != 0:
#         acc = TP.sum() / (TP + FN).sum()
#     else:
#         acc = math.nan

#     for i in range(num_classes):
#         if TP[i] + FP[i] != 0:
#             Prec[i] = TP[i] / (TP[i] + FP[i])
#             cnt_Prec += 1
#         else:
#             Prec[i] = 0

#         if TP[i] + FN[i] != 0:
#             Recall[i] = TP[i] / (TP[i] + FN[i])
#             cnt_Recall += 1
#         else:
#             Recall[i] = 0

#         if TP[i] + 1 / 2 * (FP[i] + FN[i]) != 0:
#             F1[i] = TP[i] / (TP[i] + 1 / 2 * (FP[i] + FN[i]))
#             cnt_F1 += 1
#         else:
#             F1[i] = 0

#     prec = Prec.sum() / cnt_Prec if cnt_Prec != 0 else math.nan
#     recall = Recall.sum() / cnt_Recall if cnt_Recall != 0 else math.nan
#     f1 = F1.sum() / cnt_F1 if cnt_F1 != 0 else math.nan

#     # return acc, Prec.mean(), Recall.mean(), F1.mean()
#     return acc * 100, prec * 100, recall * 100, f1 * 100

def get_metrics(TP, FP, FN):
    num_classes = TP.shape[0]
    Prec = torch.zeros(num_classes)
    Recall = torch.zeros(num_classes)
    F1 = torch.zeros(num_classes)

    cnt_Prec = 0
    cnt_Recall = 0
    cnt_F1 = 0


    try:
        acc = TP.sum() / (TP + FN).sum()
    except:
        acc = math.nan

    for i in range(num_classes):
        if TP[i] + FP[i] != 0:
            Prec[i] = TP[i] / (TP[i] + FP[i])
            cnt_Prec += 1
        else:
            Prec[i] = 0

        if TP[i] + FN[i] != 0:
            Recall[i] = TP[i] / (TP[i] + FN[i])
            cnt_Recall += 1
        else:
            Recall[i] = 0

        if TP[i] + 1 / 2 * (FP[i] + FN[i]) != 0:
            F1[i] = TP[i] / (TP[i] + 1 / 2 * (FP[i] + FN[i]))
            cnt_F1 += 1
        else:
            F1[i] = 0

    prec = Prec.sum() / cnt_Prec if cnt_Prec != 0 else math.nan
    recall = Recall.sum() / cnt_Recall if cnt_Recall != 0 else math.nan
    f1 = F1.sum() / cnt_F1 if cnt_F1 != 0 else math.nan

    # return acc, Prec.mean(), Recall.mean(), F1.mean()
    return acc * 100, prec * 100, recall * 100, f1 * 100

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger():
    def __init__(self, logfile):
        if os.path.exists(os.path.dirname(logfile)) != True:
            os.makedirs(os.path.dirname(logfile))
        
        while os.path.exists(logfile):
            time.sleep(1)
            
        file = open(logfile, 'w')
        file.close()
        
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=self.logfile
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            # print(msg % args)
            self.logger.info(msg, *args)
        else:
            # print(msg)
            self.logger.info(msg)