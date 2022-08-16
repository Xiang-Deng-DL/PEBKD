from __future__ import print_function, division

import sys
import time
import torch
from .util import AverageMeter, accuracy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


def freegrad(model_list, state):
    for param in model_list.parameters():
        param.requires_grad = state


def update_mix(model_s, model_t, train_loader, criterion_div, opt):
    a = opt.alphas
    ps = opt.ps
    
    maxdifs = 0.0
    
    model_s.eval()
    model_t.eval()
    
    for i in range(len(a)):
        for j in range(len(ps)):
            
            if i==0 and j==0: continue#too similar to the original images
            
            alpha = a[i]
            patch_size = ps[j]
            
            difs = 0.0
            
            for idx, data in enumerate(train_loader):
                
                input, target = data
                input = input.float()
                
                if torch.cuda.is_available():
                    
                    input = input.cuda()
                    mix_input = mixup_patch(input, alpha, patch_size, use_cuda=True)
                    
                    with torch.no_grad():
                        mixlogit_s = model_s(mix_input, is_feat=False, preact=False)
                        mixlogit_t = model_t(mix_input, is_feat=False, preact=False)
                        
                    mixloss_div = criterion_div(mixlogit_s, mixlogit_t)  
                    difs += mixloss_div.item()
            print('difs: %f' %difs)            
            if  difs >  maxdifs:
                maxdifs = difs
                best_a = alpha
                best_ps = patch_size
    model_s.train()
    return best_a, best_ps


def train_mixdistill(epoch, train_loader, module_list, criterion_list, optimizer, alpha, patch_size, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()
    

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()

    for idx, data in enumerate(train_loader):

        input, target = data
        data_time.update(time.time() - end)

        input = input.float()
       
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        
        p = np.random.uniform()
        
        # ===================forward=====================
        preact = False
        if p > opt.pro:

            mix_input = mixup_patch(input, alpha, patch_size, use_cuda=True)
            
            mixlogit_s = model_s(mix_input, is_feat=False, preact=preact)
            
            with torch.no_grad(): 
                mixlogit_t = model_t(mix_input, is_feat=False, preact=preact)
            mixloss_div = criterion_div(mixlogit_s, mixlogit_t)
            loss = mixloss_div
            
        else:
            logit_s = model_s(input, is_feat=False, preact=preact)
            with torch.no_grad():         
                logit_t = model_t(input, is_feat=False, preact=preact)
            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)
            loss = opt.alpha*loss_cls + opt.beta*loss_div
            
            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if p <= opt.pro and idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg





def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def mixup_patch(x, alpha, path_size, use_cuda=True,):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    
    #alpha = opt.mix_alpha
    #path_size = opt.patch_size
    
    batch_size, channel, h, w = x.shape
    path_num = int(h/path_size)*int(w/path_size)
 
    #lam = np.random.beta(alpha, alpha, size=(batch_size, path_num) )##64, 16, 1, 1, 1
    
    lam = np.random.beta(alpha, alpha, size=(1, path_num) )
    
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
        lam= torch.tensor(lam).cuda()
        lam = lam.float()
    else:
        index = torch.randperm(batch_size)
        lam= torch.tensor(lam)
        lam = lam.float()
    
    x = x.view(batch_size, channel, int(h/path_size), path_size, int(w/path_size), path_size)#64, 3, 4, 8, 4, 8
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()#64, 3, 4, 4, 8, 8
    x = x.view(batch_size, channel, path_num, path_size, path_size)# 64, 3, 16, 8, 8
    x = x.permute(0, 2, 1, 3, 4).contiguous()# 64,16, 3, 8, 8
    #lam = lam.reshape(batch_size, path_num, 1, 1, 1)
    lam = lam.reshape(1, path_num, 1, 1, 1)

    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    mixed_x = mixed_x.permute(0, 2, 1, 3, 4).contiguous()#64,3, 16, 8, 8
    mixed_x = mixed_x.view(batch_size, channel, int(h/path_size), int(w/path_size), path_size, path_size)#64, 3, 4, 4, 8, 8
    mixed_x = mixed_x.permute(0, 1, 2, 4, 3, 5).contiguous()#64, 3, 4, 8, 4, 8
    mixed_x = mixed_x.view(batch_size, channel, h, w)
    
    return mixed_x
