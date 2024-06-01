import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
from pdb import set_trace as brk
import time

from pdb import set_trace as brk
from headline_data_set import HeadlineDataset

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, filter_h, out_channels, max_length, filter_d=300, in_channels=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(filter_h, filter_d)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_length,1)))
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        return out
    
class CUE_CNN(nn.Module):
    def __init__(self, filters, out_channels, max_length, hidden_units, drop_prob, num_classes=2):
        super(CUE_CNN, self).__init__()
        self.conv1 = ConvNet(filters[0], out_channels=out_channels, max_length=max_length - filters[0]  + 1)
        self.conv2 = ConvNet(filters[1], out_channels=out_channels, max_length=max_length - filters[1]  + 1)
        self.conv3 = ConvNet(filters[2], out_channels=out_channels, max_length=max_length - filters[2]  + 1)
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat((out1, out2, out3), dim=1)
        return out

class MixtureOfExperts(nn.Module):
    def __init__(self, filters, out_channels, max_length, hidden_units, drop_prob, lstm_input_size, hidden_size_lstm, hidden_units_attention, pretrained_weight, num_classes=2):
        super(MixtureOfExperts, self).__init__()
        #brk()
        self.embed = nn.Embedding(pretrained_weight.shape[0], pretrained_weight.shape[1])
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        
        self.cue_cnn = CUE_CNN(filters, out_channels, max_length, hidden_units, drop_prob, num_classes)
        self.bi_lstm = nn.LSTM(lstm_input_size, hidden_size_lstm, num_layers=1, bidirectional=True)
        
        self.attention_mlp = nn.Linear(hidden_size_lstm * 2, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 3 + hidden_size_lstm * 2, hidden_units),
            nn.Tanh(), 
            nn.Dropout(drop_prob), #dropout
            nn.Linear(hidden_units, 50),
            nn.Tanh(), 
            nn.Dropout(drop_prob), #dropout
            nn.Linear(50, num_classes))
    
    def forward(self, y, sents=None, vis_attention=False):
        x = self.embed(y)
        out1 = self.cue_cnn(x.unsqueeze(1))
        out2 = self.bi_lstm(x.transpose(0,1))[0].transpose(0,1)
        out3 = self.attention_mlp(out2)
        if vis_attention:
            z = out3.view(x.size(0), x.size(1))
            for i in range(z.size(0)):
                w = z[i][7:7+len(sents[i].split())]
                att = nn.Softmax()(w * 10000).data.cpu().numpy()
                self.showAttention(sents[i], ["attention"], att)

        out4 = torch.mul(nn.Softmax()(out3.view(x.size(0), x.size(1))).unsqueeze(2).repeat(1,1,out2.size(2)), out2)
        out5 = torch.sum(out4, dim=1)
        out = torch.cat((out1, out5), dim=1)
        out = self.mlp(out)
        return out
    
    def showAttention(self, input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.reshape(1, -1), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' '), rotation=90)
        ax.set_yticklabels([''] + output_words + [''])

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.savefig('progress/vis_attention__'+ input_sentence +'__.jpg')
        plt.clf()

parser = argparse.ArgumentParser(description='PyTorch CUE CNN Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=128, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-v', '--vis_att', dest='vis_attention', action='store_true',
                    help='visualize attention on validation set')

best_prec1 = 0

def plot_stats(epoch, data_1, data_2, data_3, label_1, label_2, label_3, plt, ylabel):
    data_1_cpu = torch.tensor(data_1).cpu().detach().numpy()
    data_2_cpu = torch.tensor(data_2).cpu().detach().numpy()
    data_3_cpu = torch.tensor(data_3).cpu().detach().numpy()
    plt.plot(range(epoch), data_1_cpu, 'r--', label=label_1)
    plt.plot(range(epoch), data_2_cpu, 'g--', label=label_2)
    plt.plot(range(epoch), data_3_cpu, 'b--', label=label_3)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend()


def main():
    run_time = time.ctime().replace(' ', '_')[:-8] 
    directory = 'progress/' + run_time
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory + '/logs.txt', 'w',encoding='utf-8')
    f1 = open(directory + '/vis.txt', 'w',encoding='utf-8')
    global args, best_prec1
    print ("GPU processing available : ", torch.backends.mps.is_available())
    # print ("Number of GPU units available :", torch.mps.device_count())
    args = parser.parse_args()

    ## READ DATA
    filter_h = [4,6,8]
    
    train_sampler = None 
    train_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_train.txt', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        pad = max(filter_h) - 1,
        whole_data='DATA/txt/headlines_clean.txt',
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)
    

    val_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_val.txt', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        pad = max(filter_h) - 1,
        word_idx = train_dataset.word_idx,
        pretrained_embs = train_dataset.pretrained_embs,
        max_l=train_dataset.max_l,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True)
    
    test_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_test.txt', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        pad = max(filter_h) - 1,
        word_idx = train_dataset.word_idx,
        pretrained_embs = train_dataset.pretrained_embs,
        max_l=train_dataset.max_l,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True)

    parameters = {"filters": filter_h,
                  "out_channels": 100,                  
                  "max_length": train_dataset.max_l + 2  * (max(filter_h) - 1),
                  "hidden_units": 64,
                  "drop_prob": 0.2,
                  "user_size": 400,
                  "epochs":args.epochs}
    
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MixtureOfExperts(parameters['filters'], parameters['out_channels'], parameters['max_length'], parameters['hidden_units'], 
                    parameters['drop_prob'], 300, 256, 128, train_dataset.pretrained_embs)
    # model = torch.nn.DataParallel(model).cuda()
    device = torch.device("mps")
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr = args.lr,
                                     rho=args.momentum,
                                     weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    train_prec1_plot = []
    train_loss_plot = []
    val_prec1_plot = []
    val_loss_plot = []
    test_prec1_plot = []
    test_loss_plot = []

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_prec1_plot = train_prec1_plot + checkpoint['train_prec1_plot']
            train_loss_plot = train_loss_plot + checkpoint['train_loss_plot']
            val_prec1_plot = val_prec1_plot + checkpoint['val_prec1_plot']
            val_loss_plot = val_loss_plot + checkpoint['val_loss_plot']
            test_prec1_plot = test_prec1_plot + checkpoint['test_prec1_plot']
            test_loss_plot = test_loss_plot + checkpoint['test_loss_plot']
            model.load_state_dict(checkpoint['state_dict'])
            word_idx = checkpoint['word_idx']
            train_dataset.word_idx = word_idx
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_val.txt', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        pad = max(filter_h) - 1,
        word_idx = train_dataset.word_idx,
        pretrained_embs = train_dataset.pretrained_embs,
        max_l=train_dataset.max_l,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True)
    
    test_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_test.txt', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        pad = max(filter_h) - 1,
        word_idx = train_dataset.word_idx,
        pretrained_embs = train_dataset.pretrained_embs,
        max_l=train_dataset.max_l,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate and args.vis_attention:
        validate(test_loader, model, criterion, f, f1, tag='test', vis_attention=True)
        return
    
    if args.evaluate:
        validate(test_loader, model, criterion, f, f1, tag='test')
        return
    
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train_prec1, train_loss  = train(train_loader, model, criterion, optimizer, epoch, f)
        train_prec1_plot.append(train_prec1)
        train_loss_plot.append(train_loss)
        
        # evaluate on validation set
        val_prec1, val_loss = validate(val_loader, model, criterion, f, f1, tag='val')
        val_prec1_plot.append(val_prec1)
        val_loss_plot.append(val_loss)
        
        # evaluate on test set
        test_prec1,test_loss = validate(test_loader, model, criterion, f, f1, tag='test')
        test_prec1_plot.append(test_prec1)
        test_loss_plot.append(test_loss)
        
        f1.close()
        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        save_checkpoint({
            'train_prec1_plot':train_prec1_plot,
            'train_loss_plot':train_loss_plot,
            'val_prec1_plot':val_prec1_plot,
            'val_loss_plot':val_loss_plot,
            'test_prec1_plot':test_prec1_plot,
            'test_loss_plot':test_loss_plot,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'word_idx' : train_dataset.word_idx,
        }, is_best)
        
        #plot data
        font = {'size' : 18}
        matplotlib.rc('font', **font)
        plt.figure(figsize=(12,12))
        plt.subplot(2,1,1)
        plot_stats(epoch+1, train_loss_plot, val_loss_plot, test_loss_plot, 'train_loss', 'val_loss', 'test_loss', plt, 'Loss')
        plt.subplot(2,1,2)
        plot_stats(epoch+1, train_prec1_plot, val_prec1_plot, test_prec1_plot, 'train_acc', 'val_acc', 'test_acc', plt, 'Accuracy')
        plt.savefig('progress/' + run_time + '/stats.jpg')
        plt.clf()
    f.close()

def train(train_loader, model, criterion, optimizer, epoch, f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    device = torch.device("mps")
    # model.to(device)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, sent) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # if torch.cuda.is_available():
        #     target = target.cuda()
        #     input = input.cuda()

        input = torch.autograd.Variable(input).type(torch.LongTensor)
        target = torch.autograd.Variable(target)

        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress_stats = 'Time: {0} Epoch: [{1}][{2}/{3}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   time.ctime()[:-8], epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1)
            print(progress_stats)
            f.write(progress_stats + "\n")
            f.flush()
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, f, f1, tag, vis_attention=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    device = torch.device("mps")

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data_points in enumerate(val_loader):
        if tag != 'semeval':
            input, target, sents = data_points
        else:
            input, user_embeddings, target, sents = data_points
        # if torch.cuda.is_available():
        #     target = target.cuda()
        #     input = input.cuda()

        input = torch.autograd.Variable(input, volatile=True).type(torch.LongTensor)
        target = torch.autograd.Variable(target, volatile=True)

        input = input.to(device)
        target = target.to(device)
    
        # compute output
        output = model(input, sents, vis_attention)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1, ))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if tag == 'test' and args.evaluate:
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            confidences = nn.Softmax()(output)
            for x in range(target.size(0)):
                progress_stats = '{label} | {pred} |{confidence} | {sent} \n'.format(label=target[x].data[0],pred=pred[0][x].data[0],confidence=max(confidences[x]).data[0],
                   sent=sents[x])
                f1.write(progress_stats)
            f1.flush()

    val_stats = '{tag}: Time {time} * Prec@1 {top1.avg:.3f} Loss {loss.avg:.4f}'.format(
        tag=tag,time=time.ctime()[:-8],top1=top1, loss=losses)
    print(val_stats)
    f.write(val_stats + "\n")
    f.flush()
    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.8** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = target.data
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    #time.sleep(10)
    return res


if __name__ == '__main__':
    main()