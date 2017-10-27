import os
import glob
import random
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.folder import *

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from graphviz import Digraph
import torch
from torch.autograd import Variable


def make_dot(var, params):
    """ From pytorch forums: https://discuss.pytorch.org/t/print-autograd-graph/692/16


    Produces Graphviz representation of PyTorch autograd graph
    
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='center',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

def get_loaders(batch_size, num_workers, imsize):
    # get number of files
    filez = glob.glob('../data/*/*.jpg')
    # Data loading code
    traindir = '../data/'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    def shear(img):
        width, height = img.size
        m = random.uniform(-0.1, 0.1)
        xshift = abs(m) * width
        new_width = width + int(round(xshift))
        img = img.transform((new_width, height), Image.AFFINE,
                            (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                            Image.BICUBIC)
        return img

    def rotate(img):
        rotation = random.uniform(-360,360)
        img = img.rotate(rotation)
        return img

    def clipped_zoom(img):

        zoom_factor = random.uniform(0.7, 1.30)

        h, w = img.size

        # width and height of the zoomed image
        zh = int(np.round(zoom_factor * h))
        zw = int(np.round(zoom_factor * w))

        # zooming out
        if zoom_factor < 1:
            # bounding box of the clip region within the output array
            top = (h - zh) // 2
            left = (w - zw) // 2
            # zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = img.resize((zw,zh))

        # zooming in
        elif zoom_factor > 1:
            # bounding box of the clip region within the input array
            top = (zh - h) // 2
            left = (zw - w) // 2
            out = img.resize((zw,zh))
            # `out` might still be slightly larger than `img` due 
            # to rounding, so trim off any extra pixels at the edges
            trim_top = ((out.size[0] - h) // 2)
            trim_left = ((out.size[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]

        # if zoom_factor == 1, just return the input array
        else:
            out = img
        return out

    # split the files into train and test 
    sample_idx = list(range(0, len(filez)))
    # get the label for each file name
    labels = []
    for fnm in filez:
        folder_name = fnm.split('data/')[1].split('/')[0]
        labels.append(folder_name)

    lblr = LabelEncoder()
    labels = lblr.fit_transform(labels)
    print('Found:', len(set(labels)), 'labels')

    # save the labelencoder to disk to retrieve the class names later
    joblib.dump(lblr, '../models/label_encoder.pkl')

    train_idx, valid_idx, train_lab, val_lab = train_test_split(sample_idx, 
                                                                labels,
                                                                test_size=0.1, 
                                                                stratify=labels,
                                                                random_state=666)


    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trn_imsize = int(imsize * 0.5) + imsize

    train_loader = data.DataLoader(
        datasets.ImageFolder(traindir,
                             transforms.Compose([
                                 transforms.Scale(trn_imsize),
                                 transforms.RandomSizedCrop(imsize),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    val_loader = data.DataLoader(
        datasets.ImageFolder(traindir,
                             transforms.Compose([
                                 transforms.Scale(imsize),
                                 transforms.CenterCrop(imsize),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader, len(train_idx)

def get_prediction_loader(num_workers):
    test_loader = data.DataLoader(
        ImageFolderPredict('../data/test'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return test_loader


# plot the loss curves
def plot_losses(model_name, train_losses, valid_losses):
    plt.plot(train_losses, linewidth=2, label='train loss')
    plt.plot(valid_losses, linewidth=2, label='valid loss')
    plt.legend(loc=2)
    plt.savefig('../plots/{}_train_curves.png'.format(args.model_name), dpi=800)
    #plt.show()

def load_log(model_nm, exp_nm, load_best=True):

    if os.path.exists('../logs/losses_{}_{}.txt'.format(model_nm, exp_nm)):
        losses = pd.read_csv('../logs/losses_{}_{}.txt'.format(model_nm, exp_nm), sep=',', header=None)
        losses.columns = ['epoch', 'trainloss', 'validloss']

        if load_best:
            min_idx = losses['validloss'].idxmin()
            start_epoch = losses['epoch'][min_idx] + 1
            best_val_loss = losses['validloss'][min_idx]
        else:
            start_epoch = losses['epoch'][losses.index[-1]] + 1
            best_val_loss = losses['validloss'][losses.index[-1]]

    else:
        start_epoch = 0
        best_val_loss = 10.

    return start_epoch, best_val_loss

