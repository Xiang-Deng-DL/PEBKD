from __future__ import print_function

import os
import argparse
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate
from helper.loops import train_mixdistill as train, validate, update_mix
from distiller_zoo import DistillKL


def parse_option():


    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=500, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=240, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--pro', type=float, default=0.5, help='pro of training samples')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    #mixpatch
    parser.add_argument('--patch_size', type=int, default=16, help='patch size')
    parser.add_argument('--mix_alpha', type=float, default=1.0, help='patch size')
    parser.add_argument('--updata_epoch', type=int, default=50, help='updata_epoch')
    
    parser.add_argument('--alphas', type=str, default='0.1, 0.5, 1', help='alphas')
    parser.add_argument('--ps', type=str, default='32, 16', help='patch_sizes')
    parser.add_argument('--search_T', type=float, default=1, help='temperature for search')
    
    
    # model
    parser.add_argument('--model_s', type=str, default='resnet20')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    
    # distillation
    parser.add_argument('--distill', type=str, default='PE', choices=['PE'])

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--alpha', type=float, default=0.1, help='cofficients for CE loss')
    parser.add_argument('--beta', type=float, default=0.9, help='cofficients for KL loss')    

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'
        
    opt.updata_epoch = int( opt.updata_epoch/opt.pro )

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        point = int( int(it)/opt.pro )
        opt.lr_decay_epochs.append(point)

    alps = opt.alphas.split(',')
    opt.alphas = list([])
    for alp in alps:
        opt.alphas.append(float(alp))
    
    patchs = opt.ps.split(',')
    opt.ps = list([])
    for p_s in patchs:
        opt.ps.append(int(p_s))
        
    teache_name = opt.path_t.split('/')[-1].split('.')[0]

    opt.model_name = '{}_{}_{}_lr_{}_pro_{}_decay_{}_trial_{}'.format(opt.model_s, teache_name, opt.dataset, opt.learning_rate, opt.pro,
                                                            opt.weight_decay, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1].split('.')
    return segments[0]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    
    model = model_dict[model_t](num_classes=n_cls)
    if model_t=='resnet110':
        model.load_state_dict(torch.load(model_path)['model'])
    else:
        model.load_state_dict(torch.load(model_path)['state_dict'])#['model']
    print('==> done')
    return model

def main():
    best_acc = 0

    opt = parse_option()
    opt.epochs=int(opt.epochs/opt.pro)
    

    print(opt)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_search = DistillKL(opt.search_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation


    # optimizer
    optimizer = optim.SGD(model_s.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        criterion_search.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        
        if epoch==1:     
            alpha = opt.mix_alpha
            patch_size = opt.patch_size
            print('epoch: %d, alpha: %f, patch_size: %d' %(epoch, alpha, patch_size))
        elif epoch%opt.updata_epoch==0:
            alpha, patch_size = update_mix(model_s, model_t, train_loader, criterion_search, opt)
            print('epoch: %d, alpha: %f, patch_size: %d' %(epoch, alpha, patch_size))

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, alpha, patch_size, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model_s': model_s.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper is from the last epoch.
    print( 'best accuracy:', best_acc.cpu().numpy() )

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
