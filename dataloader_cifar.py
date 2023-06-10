from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def extract_data(filename,TRAIN_NUM):

    LABEL_SIZE, IMAGE_SIZE = 1, 32
    NUM_CHANNELS = 3

    bytestream=open(filename,'rb') 
    buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS+LABEL_SIZE))
    data = np.frombuffer(buf, dtype=np.uint8)

    data = data.reshape(TRAIN_NUM,LABEL_SIZE+IMAGE_SIZE* IMAGE_SIZE* NUM_CHANNELS)
  
    labels_images = np.hsplit(data, [LABEL_SIZE])

    images = labels_images[1].reshape((TRAIN_NUM, 3, 32, 32))
    images = images.transpose((0, 2, 3, 1))
    labels = labels_images[0].reshape(TRAIN_NUM)

    return labels, images

def extract_data_cifar100(filename,TRAIN_NUM):

    C_LABEL_SIZE, LABEL_SIZE, IMAGE_SIZE = 1, 1, 32
    NUM_CHANNELS = 3

    bytestream=open(filename,'rb') 
    buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS+LABEL_SIZE+C_LABEL_SIZE))
    data = np.frombuffer(buf, dtype=np.uint8)

    data = data.reshape(TRAIN_NUM,C_LABEL_SIZE+LABEL_SIZE+IMAGE_SIZE* IMAGE_SIZE* NUM_CHANNELS)
  
    labels_images = np.hsplit(data, [LABEL_SIZE+C_LABEL_SIZE])
    labels_ = np.hsplit(labels_images[0], [C_LABEL_SIZE])

    images = labels_images[1].reshape((TRAIN_NUM, 3, 32, 32))
    images = images.transpose((0, 2, 3, 1))
    
    labels = labels_[1].reshape(TRAIN_NUM)

    return labels, images

def extract_index_data(filename,TRAIN_NUM):

    INDEX_SIZE, LABEL_SIZE, IMAGE_SIZE = 4, 1, 32
    NUM_CHANNELS = 3

    bytestream=open(filename,'rb') 
    buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS+LABEL_SIZE+INDEX_SIZE))
    data = np.frombuffer(buf, dtype=np.uint8)

    data = data.reshape(TRAIN_NUM,INDEX_SIZE+LABEL_SIZE+IMAGE_SIZE* IMAGE_SIZE* NUM_CHANNELS)
  
    labels_images = np.hsplit(data, [LABEL_SIZE+INDEX_SIZE])
    labels_index = np.hsplit(labels_images[0], [INDEX_SIZE])

    images = labels_images[1].reshape((TRAIN_NUM, 3, 32, 32))
    images = images.transpose((0, 2, 3, 1))
    
    labels = labels_index[1].reshape(TRAIN_NUM)

    return labels, images
            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            if dataset=='cifar10':   
                labels, images = extract_data('%s/test_batch.bin'%root_dir,10000)
                self.test_data = images
                self.test_label = labels   
                self.test_label=torch.tensor(self.test_label, dtype=torch.long)    
            elif dataset=='cifar100':
                labels, images = extract_data_cifar100('%s/test.bin'%root_dir,10000)
                self.test_data = images
                self.test_label = labels   
                self.test_label=torch.tensor(self.test_label, dtype=torch.long)                           
        else:    
            train_data=[]
            train_label=[]
            noise_label=[]

            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d.bin'%(root_dir,n)
                    labels, images = extract_data(dpath,10000)
                    train_data.append(images)
                    train_label.append(labels)
                    
                    if noise_mode == 'sym':
                        noise_dpath = '%s/data_batch_%d_symnoise_%.2f_with_index.bin'%(root_dir,n,self.r)
                    elif noise_mode == 'asym':
                        noise_dpath = '%s/data_batch_%d_noise_%.2f_with_index.bin'%(root_dir,n,self.r)

                    nlabels, _ =  extract_index_data(noise_dpath,10000)
                    noise_label.append(nlabels)
                    
                train_data = np.concatenate(train_data)
                train_label = np.concatenate(train_label)
                noise_label = np.concatenate(noise_label)

            elif dataset=='cifar100':    
                dpath = '%s/train.bin'%root_dir
                train_label, train_data = extract_data_cifar100(dpath,50000)
                
                if noise_mode == 'sym':
                    noise_dpath = '%s/train_symnoise_%.2f_with_index.bin'%(root_dir,self.r)
                elif noise_mode == 'pair':
                    noise_dpath = '%s/train_pairnoise_%.2f_with_index.bin'%(root_dir,self.r)

                noise_label, _ =  extract_index_data(noise_dpath,50000)  

            train_label=torch.tensor(train_label, dtype=torch.long) 
            noise_label=torch.tensor(noise_label, dtype=torch.long)  
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Number of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)   

 
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader  


