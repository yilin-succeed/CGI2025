import argparse
import os
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from model import create_model
from utils import get_TP_FP_FN, get_metrics, AverageMeter, Logger

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch Training')
# train configs
parser.add_argument('--epochs', default=75, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.0008, type=float)
parser.add_argument('--gamma', default=0.987, type=float)
parser.add_argument('--num_classes', default=3, type=int)
# common configs
parser.add_argument('--dataset', default='com', type=str)
parser.add_argument('--data_path', default='../data/datasets/samm_aligned1', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument('--is_of', default=True, type=str2bool)
parser.add_argument('--is_motion', default=True, type=str2bool)

args = parser.parse_args()

device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

# CASME2
LOSO_casme2 = ['casme2' + '_' + str(i) for i in range(1, 27)]
# SAMM
LOSO_samm = ['samm' + '_' + str(i) for i in range(6, 38)]
# SMIC
LOSO_smic = ['smic' + '_' + 's' + str(i) for i in range(1, 21)]

if args.dataset == 'casme2_3':
    LOSO = LOSO_casme2
    args.data_path = '../data/datasets/casme2/'
elif args.dataset == 'samm_3':
    LOSO = LOSO_samm
    args.data_path = '../data/datasets/samm_aligned1/'
elif args.dataset == 'smic_3':
    LOSO = LOSO_smic
    args.data_path = '../data/datasets/SMIC-HC/'
elif args.dataset == 'com':
    LOSO = LOSO_casme2 + LOSO_samm + LOSO_smic
    args.data_path = '../data/datasets/samm_aligned1/'

def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    TP = torch.zeros(args.num_classes)
    FP = torch.zeros(args.num_classes)
    FN = torch.zeros(args.num_classes)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description(f'Training   [{epoch}/{args.epochs}]')

    model.train()
    for i, (images_on, images_apex, labels, of_u, of_v) in pbar:

        images_on = images_on.to(device)
        images_apex = images_apex.to(device)
        labels = labels.to(device)
        of_u = of_u.to(device)
        of_v = of_v.to(device)

        outputs = model(images_on, images_apex, of_u, of_v)
       
        loss = criterion(outputs, labels).mean()
        losses.update(loss.item(), images_on.size(0))
        tp, fp, fn = get_TP_FP_FN(outputs.detach(), labels.detach())
        TP += tp
        FP += fp
        FN += fn
        acc, prec, recall, f1 = get_metrics(TP, FP, FN)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pbar.set_postfix_str('Loss: %.2f, Acc: %.2f, Prec: %.2f, Recall: %.2f, F1: %.2f' % (losses.avg, acc, prec, recall, f1))
        pbar.set_postfix_str('Loss: %.2f, Recall: %.2f, F1: %.2f' % (losses.avg, recall, f1))
    
    return losses.avg, acc, prec, recall, f1, TP, FP, FN

def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    TP = torch.zeros(args.num_classes)
    FP = torch.zeros(args.num_classes)
    FN = torch.zeros(args.num_classes)

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    pbar.set_description(f'Validation [{epoch}/{args.epochs}]')

    model.eval()
    with torch.no_grad():
        for i, (images_on, images_apex, labels, of_u, of_v) in pbar:

            images_on = images_on.to(device)
            images_apex = images_apex.to(device)
            labels = labels.to(device)
            of_u = of_u.to(device)
            of_v = of_v.to(device)

            outputs = model(images_on, images_apex, of_u, of_v)
        
            loss = criterion(outputs, labels).mean()
            losses.update(loss.item(), images_on.size(0))
            tp, fp, fn = get_TP_FP_FN(outputs.detach(), labels.detach())
            TP += tp
            FP += fp
            FN += fn
            acc, prec, recall, f1 = get_metrics(TP, FP, FN)

            # pbar.set_postfix_str('Loss: %.2f, Acc: %.2f, Prec: %.2f, Recall: %.2f, F1: %.2f' % (losses.avg, acc, prec, recall, f1))
            pbar.set_postfix_str('Loss: %.2f, Recall: %.2f, F1: %.2f' % (losses.avg, recall, f1))

    return losses.avg, acc, prec, recall, f1, TP, FP, FN

val_TP_all = torch.zeros(args.num_classes)
val_FP_all = torch.zeros(args.num_classes)
val_FN_all = torch.zeros(args.num_classes)
val_TP_casme2 = torch.zeros(args.num_classes)
val_FP_casme2 = torch.zeros(args.num_classes)
val_FN_casme2 = torch.zeros(args.num_classes)
val_TP_samm = torch.zeros(args.num_classes)
val_FP_samm = torch.zeros(args.num_classes)
val_FN_samm = torch.zeros(args.num_classes)
val_TP_smic = torch.zeros(args.num_classes)
val_FP_smic = torch.zeros(args.num_classes)
val_FN_smic = torch.zeros(args.num_classes)

f1_scores = {}

print(args)

logger = Logger('./results/log-'+args.dataset+'-'+time.strftime('%b%d_%H-%M-%S')+'.txt')
logger.info('----------------------------------------------------------------')
logger.info(args)

for subject in LOSO:
    print('----------------------------------------------------------------')
    print(f'Subject No. {subject}\n')

    print('Load Data...')
    train_loader, val_loader = get_dataloaders(dataset=args.dataset, data_path=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers, num_loso=subject)

    print('Load Model...\n')
    model = create_model(num_classes=args.num_classes, is_of=args.is_of, is_motion=args.is_motion).to(device)

    print(f'Train Set Size: {train_loader.dataset.__len__()}')
    print(f'Validation Set Size: {val_loader.dataset.__len__()}\n')

    if val_loader.dataset.__len__() == 0:
        continue

    str_dataset = subject.split('_')[0]

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    max_val_acc = 0
    max_val_prec = 0
    max_val_recall = 0
    max_val_f1 = 0
    max_val_TP = torch.zeros(args.num_classes)
    max_val_FP = torch.zeros(args.num_classes)
    max_val_FN = torch.zeros(args.num_classes)
    best_epoch = 0

    print('Start Training.')
    for epoch in range(1, args.epochs+1):
        print('--------------------------------')
        print(f'Subject No. {subject}\n')
        print('Epoch: %d, Learning Rate: %f' % (epoch, optimizer.param_groups[0]['lr']))
        train_loss, train_acc, train_prec, train_recall, train_f1, train_TP, train_FP, train_FN = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, val_acc, val_prec, val_recall, val_f1, val_TP, val_FP, val_FN = validate(val_loader, model, criterion, epoch)

        if val_f1 > max_val_f1:
            max_val_acc = val_acc
            max_val_prec = val_prec
            max_val_recall = val_recall
            max_val_f1 = val_f1
            max_val_TP = val_TP
            max_val_FP = val_FP
            max_val_FN = val_FN
            best_epoch = epoch

        if val_acc == 100.0:
            break

        if epoch <= 50:
            scheduler.step()

        print("\nBest F1 Score: %.2f (%d)" % (max_val_f1, best_epoch))

    val_TP_all += max_val_TP
    val_FP_all += max_val_FP
    val_FN_all += max_val_FN

    if str_dataset == 'casme2':
        val_TP_casme2 += max_val_TP
        val_FP_casme2 += max_val_FP
        val_FN_casme2 += max_val_FN
    elif str_dataset == 'samm':
        val_TP_samm += max_val_TP
        val_FP_samm += max_val_FP
        val_FN_samm += max_val_FN
    elif str_dataset == 'smic':
        val_TP_smic += max_val_TP
        val_FP_smic += max_val_FP
        val_FN_smic += max_val_FN

    val_acc_all, val_prec_all, val_recall_all, val_f1_all = get_metrics(val_TP_all, val_FP_all, val_FN_all)
    val_acc_casme2, val_prec_casme2, val_recall_casme2, val_f1_casme2 = get_metrics(val_TP_casme2, val_FP_casme2, val_FN_casme2)
    val_acc_samm, val_prec_samm, val_recall_samm, val_f1_samm = get_metrics(val_TP_samm, val_FP_samm, val_FN_samm)
    val_acc_smic, val_prec_smic, val_recall_smic, val_f1_smic = get_metrics(val_TP_smic, val_FP_smic, val_FN_smic)

    print('--------------------------------')
    print(f'Subject No. {subject}\n')
    print("Subject Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (max_val_acc, max_val_prec, max_val_recall, max_val_f1))
    print(f"Overall Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (val_acc_all, val_prec_all, val_recall_all, val_f1_all))

    print(f"\nCASME2 Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (val_acc_casme2, val_prec_casme2, val_recall_casme2, val_f1_casme2))
    print(f"SAMM Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (val_acc_samm, val_prec_samm, val_recall_samm, val_f1_samm))
    print(f"SMIC Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (val_acc_smic, val_prec_smic, val_recall_smic, val_f1_smic))

    print("\nF1 Scores of Subjects")
    f1_scores[subject] = (val_loader.dataset.__len__(), max_val_f1, best_epoch)
    for (key, val) in f1_scores.items():
        print("Subject No. %s (%d): %.2f (%d)" % (key, val[0], val[1], val[2]))

# record all results
logger.info("Subject Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (max_val_acc, max_val_prec, max_val_recall, max_val_f1))
logger.info(f"Overall Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (val_acc_all, val_prec_all, val_recall_all, val_f1_all))
logger.info('')
logger.info(f"CASME2 Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (val_acc_casme2, val_prec_casme2, val_recall_casme2, val_f1_casme2))
logger.info(f"SAMM Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (val_acc_samm, val_prec_samm, val_recall_samm, val_f1_samm))
logger.info(f"SMIC Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f" % (val_acc_smic, val_prec_smic, val_recall_smic, val_f1_smic))
logger.info('')
logger.info("F1 Scores of Subjects")
f1_scores[subject] = (val_loader.dataset.__len__(), max_val_f1, best_epoch)
for (key, val) in f1_scores.items():
    logger.info("Subject No. %s (%d): %.2f (%d)" % (key, val[0], val[1], val[2]))