import sys
import math
import time
import gzip
import glob
import pickle
import argparse
import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models import VGGlike_vanilla, VGGlike_upsample
from utils import get_loaders, load_log

parser = argparse.ArgumentParser(description='Kaggle Cdiscounts Training')
parser.add_argument('--gpu', default=1, type=int, 
                    help='which gpu to run')
parser.add_argument('--batch_size', default=128, type=int, 
                    help='size of batches')
parser.add_argument('--epochs', default=500, type=int, 
                    help='number of epochs')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--es_patience', default=3, type=int, 
                    help='early stopping patience')
parser.add_argument('--lr_patience', default=1, type=int, 
                    help='learning rate decay patience')
parser.add_argument('--lr_decay_scale', default=0.1, type=float,
                    help='how much to scale learning rate on each decay')
parser.add_argument('--load_best', action='store_true', 
                    help='flag to load from checkpoint')
parser.add_argument('--load_last', action='store_true', 
                    help='flag to load from end of training')
parser.add_argument('--model_name', default='vgglike', type=str,
                    help='name of model for saving/loading weights')
parser.add_argument('--exp_name', default='vanilla', type=str,
                    help='name of experiment for saving files')
parser.add_argument('--num_workers', default=3, type=int,
                    help='how many workers to use for data loader')
parser.add_argument('--imsize', default=128, type=int,
                    help='what size to set images')
args = parser.parse_args()

# set model filenames
MODEL_CKPT = '../models/best_{}_{}_classifier.pth'.format(args.model_name, 
                                                          args.exp_name)
MODEL_FINL = '../models/last_{}_{}_classifier.pth'.format(args.model_name, 
                                                          args.exp_name)

# init some training params or load from saved
valid_patience = 0
lr_patience = 0

# load the model
if args.exp_name == 'vanilla':
    print('Using vanilla model ...')
    net = VGGlike_vanilla()
else:
    print('Using upsample model ...')
    net = VGGlike_upsample()

# load from previous run,
if args.load_best:
    print('Loading from checkpoint ...')
    #net.load_state_dict(torch.load(MODEL_CKPT))
    net.load_state_dict(torch.load('../models/best_{}_{}_classifier.pth'.format(args.model_name,
                                                                                args.exp_name)))

    # load stats from saved run
    STRT_EPOCH, best_val_loss = load_log(model_nm=args.model_name,
                                         exp_nm=args.exp_name, load_best=True)
    print('Starting from epoch {}, with best val loss {}'.format(STRT_EPOCH,
                                                                 best_val_loss))   
elif args.load_last:
    print('Loading from end of run ...')
    net.load_state_dict(torch.load(MODEL_FINL))
    # load stats from saved run
    STRT_EPOCH, best_val_loss = load_log(args.model_name, 
                                         exp_nm=args.exp_name, load_best=False)
    print('Starting from epoch {}, with best val loss {}'.format(STRT_EPOCH,
                                                                 best_val_loss))
else:
    STRT_EPOCH, best_val_loss = 0, 10.0

# cuda and GPU settings
if args.gpu == 99:
    net = torch.nn.DataParallel(net, device_ids=[0,1]).cuda()
else:
    torch.cuda.set_device(args.gpu)
    net.cuda()

cudnn.benchmark = True

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

train_loader, val_loader, len_train = get_loaders(args.batch_size, 
                                                  args.num_workers,
                                                  args.imsize)

# training loop
def train():
    net.train()
    # keep track of accuracy
    total = 0
    correct = 0
    # keep track of losses
    iter_loss = 0.
    iter_correct = 0.
    num_batch_epoch = 0
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data

        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda(async=True))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        iter_loss += loss.data[0]

        num_batch_epoch += 1
        #print('Processed batch', num_batch_epoch, labels.data)

        sys.stdout.write('\r')
        multiplier = int((float(i) / (len_train // args.batch_size)) * 10)
        sys.stdout.write('B: {:>3}/{:<3} | {:.3} | {:.3}'.format(i, 
                                                   len_train // args.batch_size,
                                                   iter_loss / num_batch_epoch, 
                                                   100.*correct/total))
        #sys.stdout.write('-' * multiplier)
        sys.stdout.flush()

    avg_loss = iter_loss / num_batch_epoch
    print('\n' + 'Train Loss: {:.3} | Train Acc: {:.3}'.format(avg_loss,
                                                        100.*correct/total))
    return iter_loss / num_batch_epoch

# validation loop
def validate():
    net.eval()
    # keep track of accuracy
    val_total = 0
    val_correct = 0
    # keep track of losses
    val_loss = 0.
    val_batch_num = 0
    for j, data in enumerate(val_loader):
        val_in, val_lab = data
        #val_in = torch.from_numpy(val_in).float()
        #val_lab = torch.from_numpy(val_lab).long()
        val_in  = Variable(val_in.cuda(), volatile=True)
        val_lab = Variable(val_lab.cuda(async=True))

        val_out = net(val_in)
        v_l = criterion(val_out, val_lab)
        val_loss += v_l.data[0]

        _, val_pred = torch.max(val_out.data, 1)
        val_total += val_lab.size(0)
        val_correct += val_pred.eq(val_lab.data).cpu().sum()

        val_batch_num += 1

    avg_vloss = float(val_loss) / val_batch_num
    print('Eval Loss: {:.3} | Eval Acc: {:.3}'.format(avg_vloss,
                                               100.*val_correct/val_total))
    return val_loss / val_batch_num, 100.*val_correct/val_total

# train the model
try:
    print('Training ...')
    train_losses = []
    valid_losses = []
    for e in range(STRT_EPOCH, args.epochs):
        print('\n' + 'Epoch {}/{}'.format(e, args.epochs))
        start = time.time()

        t_l = train()
        v_l, v_a = validate()
        train_losses.append(t_l)
        valid_losses.append(v_l)

        # write the losses to a text file
        with open('../logs/losses_{}_{}.txt'.format(args.model_name, 
                                                    args.exp_name), 'a') as logfile:
            logfile.write('{},{},{},{}'.format(e, t_l, v_l, v_a) + "\n")

        # save the model everytime we get a new best valid loss
        if v_l < best_val_loss:
            torch.save(net.state_dict(), MODEL_CKPT)
            best_val_loss = v_l
            valid_patience = 0
            lr_patience = 0

        # if the validation loss gets worse increment 1 to the patience values
        if v_l > best_val_loss:
            valid_patience += 1
            lr_patience += 1

        # if the model doesn't improve by a certain amount of epochs, 
        # lower learning rate
        if lr_patience >= args.lr_patience:
            print('Changing learning rate by {}'.format(args.lr_decay_scale))
            for params in optimizer.param_groups:
                params['lr'] = params['lr'] * args.lr_decay_scale
                lr_patience = 0
                #LR_DECAY += 5

            # start the net from the previous best 
            #net.load_state_dict(torch.load(MODEL_CKPT))

        # if the model stops improving by a certain number epochs, stop
        if valid_patience == args.es_patience:
            break

        print('Time: {}'.format(time.time()-start))

    print('Finished Training')

except KeyboardInterrupt:
    pass

torch.save(net.state_dict(), MODEL_FINL)


