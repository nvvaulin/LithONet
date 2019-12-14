import argparse
import copy
import glob
import math
import os
import os.path as osp
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import mobilenet_v2, resnet18, resnet34, resnet50
from tqdm import tqdm

from seismic.config import config, landmass_config
from seismic.models.classification.dataset import FaciesDataset
from seismic.models.classification.transform import get_augmentations


def eval_net(net, loader, device, criterion=None, writer=None,
             sample_name='test', iter=1, predict=False):
    """
        Args:
            <...>
            writer          (str) : stata writer object, e.g. tb.SummaryWriter;
            sample_name     (str) : type a name of evaluation sample (e.g. "val" or "test");
            iter            (int) : x-axis value when plotting metrics in Trains;
            predict       (boole) : whether to return predicitons (True) or stata (False, default).
    """
    if not criterion:
        criterion = torch.nn.CrossEntropyLoss()

    net.eval()
    net.to(device)

    running_loss = 0.0
    gt = np.array([])
    preds = np.array([])
    n_batches = math.ceil(len(loader.sampler) / loader.batch_size)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader, 0), desc='validation', total=n_batches):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            batch_probs = F.softmax(outputs, dim=1)
            batch_preds = batch_probs.argmax(dim=1).detach().cpu().numpy()

            gt = np.hstack((gt, labels.detach().cpu().numpy()))
            preds = np.hstack((preds, batch_preds))

            loss = criterion(outputs, labels)
            running_loss += loss.item()

    accuracy = metrics.accuracy_score(gt, preds)
    precision = metrics.precision_score(gt, preds, average='macro')
    recall = metrics.recall_score(gt, preds, average='macro')
    f1_score = metrics.f1_score(gt, preds, average='macro')

    stat = {'loss': running_loss / (i + 1), 'accuracy': float(accuracy),
            'precision': precision, 'recall': recall, 'f1_score': f1_score}

    if writer:
        writer.add_scalar(f'Loss/{sample_name}', stat['loss'], global_step=iter)
        writer.add_scalar(f'Accuracy/{sample_name}', stat['accuracy'], global_step=iter)
        writer.add_scalar(f'Precision/{sample_name}', stat['precision'], global_step=iter)
        writer.add_scalar(f'Recall/{sample_name}', stat['recall'], global_step=iter)
        writer.add_scalar(f'F1score/{sample_name}', stat['f1_score'], global_step=iter)

    if predict:
        return preds
    else:
        return stat


def train_net(net, train_loader, val_loader, device, optimizer, model_save_path=None, epochs=5,
              best_val_fscore=None, init_epoch=0, criterion=None, writer=None, lr_scheduler=None):
    if not criterion:
        criterion = torch.nn.CrossEntropyLoss()

    best_val_fscore = 0.0 if best_val_fscore is None else best_val_fscore
    best_net = None
    n_batches = math.ceil(len(train_loader.sampler) / train_loader.batch_size)

    for epoch in tqdm(range(init_epoch, epochs + init_epoch), desc='epoch'):
        net.train()
        net.to(device)
        running_loss = 0.0
        for i, batch in tqdm(enumerate(train_loader, 0), desc='training', total=n_batches):
            inputs, labels = batch[0].to(device, dtype=torch.float), batch[1].to(device, dtype=torch.long)
            optimizer.zero_grad()

            # Forward, backward, optimizerd
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Current loss
            running_loss += loss.item()

        ### Evaluate ###
        iter = epoch+1
        train_stat = eval_net(net, train_loader, device, writer=writer, sample_name="train", iter=iter)
        val_stat = eval_net(net, val_loader, device, writer=writer, sample_name="val", iter=iter)

        if lr_scheduler is not None:
            lr_scheduler.step(val_stat['accuracy'])

        print('\n[%d-th epoch] \ntrain & val metrics:\nloss\t\t%.4f & %.4f,\naccuracy\t\t%.4f & %.4f,\nprecision\t\t%.4f & %.4f,\nrecall\t\t%.4f & %.4f,\nfscore\t\t%.4f & %.4f\n' % \
              (epoch + 1, running_loss / (i + 1), val_stat['loss'],
               train_stat['accuracy'], val_stat['accuracy'], train_stat['precision'], val_stat['precision'],
               train_stat['recall'], val_stat['recall'],
               train_stat['f1_score'], val_stat['f1_score']))

        if model_save_path:
            state = {'val_fscore': val_stat['f1_score'], 'epoch': epoch, 'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict()}
            if val_stat['f1_score'] > best_val_fscore:
                writer.add_scalar('Fscore_best/val', val_stat['f1_score'], global_step=iter)
                torch.save(state, os.path.join(model_save_path,
                                               f"model_best.pkl"))
                best_val_fscore = val_stat['f1_score']
                best_net = copy.deepcopy(net)

            torch.save(state, os.path.join(model_save_path, f"model.pkl"))
            writer.add_text('CNN', 'Saving model (epoch %d)' % (epoch + 1), 0)

    print('Finished Training')
    if best_net:
        return best_net
    else:
        return net


if __name__ == "__main__":
    '''
    '''

    parser = argparse.ArgumentParser(description='Train a network for fluorographic images')

    # Arguments for stata and models saving
    parser.add_argument('--model_save_path', type=str, default=config.models_dir, help='local path for saving models')

    # Arguments for data loading and preparation
    parser.add_argument('--data_dir', type=str, default=osp.join(config.data_dir, 'LANDMASS', 'LANDMASS1'),
                        help='path to a folder with images')
    parser.add_argument('--train_file', type=str, default='train.csv',
                        help='path to a train set markup file')
    parser.add_argument('--val_file', type=str, default='val.csv',
                        help='path to a validation set markup file')
    parser.add_argument('--augmentation_intensity', type=str, default=None,
                        help='augmentation type: None, light, medium, heavy')
    parser.add_argument('--scale_image', action='store_true',
                        help='whether to use Jeremy scaling')
    parser.add_argument('--num_workers', type=int, default=2, help='num workers for data loader')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true',
                        help='whether to use only small sample of datasets')
    parser.add_argument('--test_size', type=int, default=40, help='how many samples to use in test_mode')

    # Arguments for model loading
    parser.add_argument('--model', type=str, default='resnet18', help='model name')

    # Arguments for training
    parser.add_argument('--optimizer_type', type=str, default='sgd', help='optimizer type: sgd or adamw or adam')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
    parser.add_argument('--patience', type=int, default=5, help='patience for LR scheduler')
    parser.add_argument('--gamma_factor', type=float, default=0.5, help='gamma factor for scheduler')
    parser.add_argument('--no_scheduler', dest='lr_scheduler', action='store_false',
                        help='whether to train without lr scheduler')

    # Extra parameters
    parser.add_argument('--sceptical_augmentation', action='store_true', help='do not perform flips and shifts')
    parser.add_argument('--random_state', type=int, default=24, help='random seed for random generators')

    # Arguments as "pretrained" (and "init_random_weights") and "grad_on" (and "no_grad") are boolean
    parser.set_defaults(pretrained=True, lr_scheduler=True)
    args = parser.parse_args()

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(osp.join(config.runs_dir, current_time))
    model_save_path = osp.join(args.model_save_path, args.model, current_time)
    os.makedirs(model_save_path, exist_ok=True)

    for k, v in args.__dict__.items():
        writer.add_text('CNN', f'{k}: {v}', 0)

    random.seed(args.random_state)
    os.environ['PYTHONHASHSEED'] = str(args.random_state)
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data loading
    data_dir = args.data_dir
    train_file = osp.join(data_dir, args.train_file)
    val_file = osp.join(data_dir, args.val_file)

    if args.augmentation_intensity not in [None, 'light', 'medium', 'heavy']:
        raise ValueError('Improper augmentation flag: should be equal to None, light, medium, or heavy')
    preprocess = None
    if data_dir.endswith('LANDMASS1'):
        resize = (99, 99)
    else:
        resize = (150, 300)
    augmentation = get_augmentations(augmentation_intensity=args.augmentation_intensity,
                                     sceptical_augmentation=args.sceptical_augmentation,
                                     resize=resize)
    transform = transforms.Compose(landmass_config.basic_transforms)

    ### Data loading ###
    dataset_train = FaciesDataset(data_dir, train_file, landmass_config.class_name_to_id, preprocess=preprocess, transform=transform,
                                 augmentation=augmentation, scale_image=args.scale_image, test_mode=args.test_mode, test_size=args.test_size)
    dataset_val = FaciesDataset(data_dir, val_file, landmass_config.class_name_to_id, preprocess=preprocess, transform=transform,
                                 augmentation=None, scale_image=args.scale_image, test_mode=args.test_mode, test_size=args.test_size)
    train_idx = np.arange(len(dataset_train))
    valid_idx = np.arange(len(dataset_val))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, sampler=valid_sampler,
                                             num_workers=args.num_workers)

    # Training and stata capturing process launching
    # lowercase and delete " " symbols (spacing)
    device = config.device
    net = eval(args.model)(pretrained=True)

    n_classes = len(landmass_config.class_name_to_id)
    if args.model.startswith('resnet'):
        in_feat = net.fc.in_features
        net.fc = nn.Linear(in_feat, n_classes)
    elif args.model.startswith('mobilenet'):
        in_feat = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_feat, n_classes)

    net.to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    if args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum,
            nesterov=True, weight_decay=args.weight_decay)

    if args.lr_scheduler:
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=args.gamma_factor, patience=args.patience)
    else:
        lr_scheduler = None

    val_fscore = 0.0
    init_epoch = 0

    net = train_net(net, train_loader, val_loader, device, optimizer,
                    model_save_path=model_save_path,
                    epochs=args.epochs, writer=writer,
                    best_val_fscore=val_fscore, init_epoch=init_epoch, lr_scheduler=lr_scheduler)
