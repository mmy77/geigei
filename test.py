import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model import GEInet
import numpy as np
import torch.nn as nn
import torch.optim as optim
from time  import time
import visdom
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import visdom
import random
import os
from resnet import ResNet,BasicBlock
import torch.utils.model_zoo as model_zoo
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


imgpath = '/home/lxq/GeiGait/data/'
#imgpath = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/GEI/'
root = '/home/lxq/GeiGait/'
# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))  
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        index = int(index % (self.__len__()))#train
        fn, label = self.imgs[index]
        img = self.loader(imgpath + fn)
        img = img.resize((224,224))
        imgdata = np.array(img)
        #print(imgdata.shape)
        imgdata = imgdata.swapaxes(0,2)
        #print("swap:",imgdata.shape)
        #imgdata = imgdata[np.newaxis,:]
        return imgdata,label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=root+'tumtrain.txt', transform=transforms.ToTensor())
#print(img1[0].shape,torch.tensor(img1[0]))
test_data=MyDataset(txt=root+'tumtest.txt', transform=transforms.ToTensor())


def load_data_cross(batch_size=10):
    global global_point
    #print('globale_point: ',global_point)
    data=[]
    label = []
    for batchpt in range(batch_size):
        tempdata,templabel = test_data.__getitem__(global_point)
       # print(np.max(tempdata))
        data.append(tempdata)
        label.append(templabel)
        global_point = global_point + 1
    return torch.from_numpy(np.array(data)).float(),torch.from_numpy(np.array(label)).long()

def load_model(model,pretrainedfile):
    if(pretrainedfile[-3:]=='pth'): #resnet
        model_dict = model.state_dict()
        weights=torch.load(pretrained_file)
        weight2 = {k:v for k,v in weights.items() if k in model_dict}
        model_dict.update(weight2)
        model.load_state_dict(model_dict)
    elif(pretrainedfile[-3:]=='pkl'): #pretrained
        model = torch.load(pretrainedfile)
    else:
        return model
    return model

#pretrained_file = 'resnet18-5c106cde.pth'
pretrained_file = 'model/GeiNet_tum2000.pkl'
model = ResNet(BasicBlock,[2,2,2,2],num_classes=125).cuda()
model = load_model(model,pretrained_file)

global_point = 100
print('model ok')
model = model.cuda()

start = time()

batch_size = 50


for ita in range(50):
    countall = 0
    count5all = 0
    #print("global_point:",global_point)
    data1,label1 = load_data_cross(batch_size=batch_size) #5
    data = torch.autograd.Variable(data1.cuda())
    out = model(data)
        #print('out#############\n',out)
    tout = out.cpu().detach().numpy()
    #print(out.shape,label2.shape)
    pred = []
    pred5 = []
    maxp = -1
    max_index = 0
    for i in range(batch_size):
        one_out = tout[i,:]
        #print(one_out)
        max_index = np.argsort(-one_out)[0]
        max5_index = np.argsort(-one_out)[0:5]
        pred.append(max_index)
        pred5.append(max5_index) #
    pred = np.array(pred)
    label2=np.array(label1) #answer
    
    static = pred-label2

    zero = static==0
    zero_num = static[zero]
    countall = countall + zero_num.size
    #print("label:",label2, "pred:",pred, "pred5:",pred5,"static:",static)
    for i in range(len(pred5)):
        if label2[i] in pred5[i]:
            count5all = count5all + 1
    print("top1:",countall/batch_size, " top5:",count5all/batch_size)




#test
