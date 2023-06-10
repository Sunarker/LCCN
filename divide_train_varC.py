from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import struct

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')


parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--warm_up', default=30, type=int, help='num of epochs to pretrain')
parser.add_argument('--warm_up_trans', default=50, type=int, help='num of epochs to pretrain transition matrix')
parser.add_argument('--lccn_warm', default=30, type=int)
parser.add_argument('--noise_mode',  default='asym')
parser.add_argument('--r', default=0.1, type=float, help='noise ratio')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T_trans', default=0, type=float, help='sharpening temperature for lccn')
parser.add_argument('--checkpoint_path', default='./checkpoint/test/', type=str, help='path to dataset')
parser.add_argument('--num_epochs', default=270, type=int)
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--backbone', default='ResNet18', type=str)
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--data_path', default='./data/cifar10/cifar-10-batches-bin', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_class', default=10, type=int)
args = parser.parse_args()

if args.dataset == 'cifar10':
    args.data_path = args.data_dir + 'cifar10/cifar-10-batches-bin/'
elif args.dataset == 'cifar100':
    args.data_path = args.data_dir + 'cifar100/cifar-100-binary/'


stats_path = args.checkpoint_path + '%s_%.1f_%s_%d_lamdau_%.1f_t_%.3f_lr_%.1f_T_%d_warm_%d_fix_%d_all'%(args.dataset,args.r,args.noise_mode,args.lambda_u,args.p_threshold,args.lr,args.T_trans,args.warm_up,args.warm_up+args.warm_up_trans,args.warm_up+args.num_epochs)+'_stats.txt'
test_path = args.checkpoint_path + '%s_%.1f_%s_%d_lamdau_%.1f_t_%.3f_lr_%.1f_T_%d_warm_%d_fix_%d_all'%(args.dataset,args.r,args.noise_mode,args.lambda_u,args.p_threshold,args.lr,args.T_trans,args.warm_up,args.warm_up+args.warm_up_trans,args.warm_up+args.num_epochs)+'_acc.txt'
c_path = args.checkpoint_path + '%s_%.1f_%s_%d_lamdau_%.1f_t_%.3f_lr_%.1f_T_%d_warm_%d_fix_%d_all'%(args.dataset,args.r,args.noise_mode,args.lambda_u,args.p_threshold,args.lr,args.T_trans,args.warm_up,args.warm_up+args.warm_up_trans,args.warm_up+args.num_epochs)+'_transition.txt'

stats_log=open(stats_path,'w') 
test_log=open(test_path,'w') 
c_log=open(c_path,'w') 

if args.dataset == 'cifar100' and args.noise_mode == 'asym':
    args.noise_mode = 'pair'

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def init_C(model):  

    model.eval()

    infer_z = torch.zeros(50000).cuda()
    noisy_y = torch.zeros(50000).cuda()
    C = torch.zeros((args.num_class,args.num_class)).cuda()
 
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            _, pred = torch.max(outputs, 1) 
            
            for b in range(inputs.size(0)):
                infer_z[index[b]] = pred[b] 
                noisy_y[index[b]] = targets[b] 

                C[pred[b]][targets[b]] += 1

    return infer_z, noisy_y, C

def eval_C(model,T,clean_label,epoch,all_loss):    
    
    model.eval()

    losses = torch.zeros(50000)
    C = torch.zeros((args.num_class,args.num_class)).cuda()
    correct = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs)  

            if (T == 0).any():
                _, pred = torch.max(outputs, 1)
                loss = CE(outputs, targets)
            else:
                out_probs = torch.softmax(outputs, dim=1)
                unnorm_probs = out_probs.mul(T.transpose(1,0)[targets])
                probs = unnorm_probs / torch.sum(unnorm_probs, dim=1, keepdim=True)
                sampler = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs)
                _, labels_ =  torch.max(sampler.sample(), 1) 
                loss = CE(probs, targets) 

            for b in range(inputs.size(0)):    
                losses[index[b]]=loss[b]

                if (T == 0).any():
                    C[pred[b]][targets[b]] += 1
                    if pred[b] == clean_label[int(index[b].numpy())]:
                        correct += 1
                else: 
                    C[labels_[b]][targets[b]] += 1
                    if labels_[b] == clean_label[int(index[b].numpy())]:
                        correct += 1

    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()] 

    return  prob, all_loss, C, correct


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,T):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        labels_x_int = labels_x
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)       

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            if (T == 0).any():
                px = w_x*labels_x + (1-w_x)*px               
            else:
                unnorm_probs = px.mul(T.transpose(1,0)[labels_x_int])
                probs = unnorm_probs / torch.sum(unnorm_probs, dim=1, keepdim=True)        
                px = w_x*labels_x + (1-w_x)*probs

            ptx = px**(1/args.T) # temparature sharpening     

            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()
                               
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, args.warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.warm_up+args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)    
        L = loss 
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.warm_up+args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def test_single(epoch,net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)         
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():

    if args.backbone == 'ResNet18':
        model = ResNet18(num_classes=args.num_class)
    elif args.backbone == 'ResNet34':
        model = ResNet34(num_classes=args.num_class)
    model = model.cuda()
    return model

def read_label(noise_ratio, args):

  img_label = dict()
  if args.dataset == 'cifar10':
    for i in range(1,6):
        if args.noise_mode == 'sym':
            path = args.data_path + 'data_batch_%d_symnoise_%.2f_with_index.bin'%(i,noise_ratio)
        elif args.noise_mode == 'asym':
            path = args.data_path + 'data_batch_%d_noise_%.2f_with_index.bin'%(i,noise_ratio)
        with open(path,'rb') as f:
            data = f.read(3077)
            while data:
                ind = struct.unpack('I', data[:4])
                ind = ind[0]
                label = data[4]
                img_label[ind] = label
                data = f.read(3077)
  elif args.dataset == 'cifar100':
    if args.noise_mode == 'sym':
        path = args.data_path + 'train_symnoise_%.2f_with_index.bin'%noise_ratio
    elif args.noise_mode == 'pair':
        path = args.data_path + 'train_pairnoise_%.2f_with_index.bin'%noise_ratio
    with open(path,'rb') as f:
        data = f.read(3077)
        while data:
            ind = struct.unpack('I', data[:4])
            ind = ind[0]
            label = data[4]
            img_label[ind] = label
            data = f.read(3077)
    
  return img_label

def write_c_dual(C1,C2,correct1,correct2,epoch,log):
    trans='\nEpoch:%d Correct1:%d Correct2:%d\n'%(epoch,correct1,correct2)
    trans +='C1:\n'
    for line in C1:
        for c in line:
            trans += '{:5d}'.format(int(c))
        trans += '\n'
    trans +='C2:\n'
    for line in C2:
        for c in line:
            trans += '{:5d}'.format(int(c))
        trans += '\n'
    log.write(trans)
    log.flush()


clean_label = read_label(0.00, args)

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks
T1 = torch.zeros((args.num_class,args.num_class))
T2 = torch.zeros((args.num_class,args.num_class))

for epoch in range(args.warm_up+args.num_epochs+1):   
    lr=args.lr
    if epoch >= args.lccn_warm+args.warm_up_trans-10:
        lr = args.lr/5 
    if epoch >= 200:
        lr = args.lr/10     
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<args.warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
    
    if epoch==args.lccn_warm:
        print('\nInitialize..')
        infer_z1, noisy_y1, C1 = init_C(net1)
        infer_z2, noisy_y2, C2 = init_C(net2)

        alpha = 1.0 
        T_init1 = (C1 + alpha) / torch.sum(C1 + alpha, dim=1, keepdim=True)
        T_init2 = (C2 + alpha) / torch.sum(C2 + alpha, dim=1, keepdim=True)
        T1 = T_init1
        T2 = T_init2
        write_c_dual(C1,C2,0,0,epoch,c_log) 

    if epoch>=args.warm_up:
        prob1, all_loss[0], C1, correct1 = eval_C(net1, T1, clean_label, epoch, all_loss[0])
        prob2, all_loss[1], C2, correct2 = eval_C(net2, T2, clean_label, epoch, all_loss[1]) 

        pred1 = (prob1 > args.p_threshold) 
        pred2 = (prob2 > args.p_threshold)
        
        if epoch>=args.lccn_warm+args.warm_up_trans:
            write_c_dual(C1,C2,correct1,correct2,epoch,c_log)
            T1 = (C1 + alpha) / torch.sum(C1 + alpha, dim=1, keepdim=True)
            T2 = (C2 + alpha) / torch.sum(C2 + alpha, dim=1, keepdim=True)
            if args.T_trans>0:
                T1_temp = T1**(1/args.T_trans)
                T2_temp = T2**(1/args.T_trans)
                T1 = T1_temp / torch.sum(T1_temp, dim=1, keepdim=True)
                T2 = T2_temp / torch.sum(T2_temp, dim=1, keepdim=True)
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,T1) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,T2) # train net2         

    test(epoch,net1,net2)  


