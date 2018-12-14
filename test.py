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


imgpath = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/GEI/'
#imgpath = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/GEI/'
root = '/home/xiaoqian/GeiGait/'
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

train_data=MyDataset(txt=root+'datalist/MT_train_list.txt', transform=transforms.ToTensor())
#print(img1[0].shape,torch.tensor(img1[0]))
test_data=MyDataset(txt=root+'datalist/MT_gallery_list.txt', transform=transforms.ToTensor())
probe_data=MyDataset(txt=root+'datalist/MT_probe_list.txt', transform=transforms.ToTensor())

def load_data_cross(batch_size=10,data_source = probe_data):
    global global_point
    #print('globale_point: ',global_point)
    data=[]
    label = []
    for batchpt in range(batch_size):
        if(global_point>=data_source.__len__()):
            continue
        tempdata,templabel = data_source.__getitem__(global_point)
       # print(np.max(tempdata))
        data.append(tempdata)
        label.append(templabel)
        global_point = global_point + 1
    return torch.from_numpy(np.array(data)).float().cuda(),label#torch.from_numpy(np.array(label)).long()

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
def calcul(output,result,label):#512  n*512  #numpy
    top5 = []
    mse = []
    for i in range(result.shape[0]):
        subr = output-result[i]
        mse.append(np.sum(subr**2))
    mse = np.array(mse)
    max_index = np.argsort(mse)[0]
    max5_index = np.argsort(mse)[0:5]#min
    top1 = label[max_index]
    for j in range(5):
        top5.append(label[max5_index[j]])
    return top1,top5


model = ResNet(BasicBlock,[2,2,2,2],num_classes=150).cuda()
#model = load_model(model,pretrained_file)
global_point=0

start = time()
pretrained_file = 'model/casia_MT2900.pkl'
pre_model = torch.load(pretrained_file)
pre_model = pre_model.state_dict()
model_dict = model.state_dict()
weight = {k:v for k,v in pre_model.items() if k in model_dict}
model_dict.update(weight)
model.load_state_dict(model_dict)
model.eval()
'''
data,label = load_data_cross()
result = model(data).detach().cpu()
while global_point<test_data.__len__():

    data,label_temp = load_data_cross()
    #print(global_point,data.shape,label_temp.shape)
    output = model(data).detach().cpu()
    label = torch.cat((label,label_temp),0)
    result = torch.cat((result,output),0)#10,512
result = result.numpy()
label = label.numpy()
np.save("model/MT_galler_list.npy",result)
np.save("model/MT_galler_label.npy",label)
'''
print('gallery load')
result = np.load("model/MT_galler_list.npy")
label = np.load("model/MT_galler_label.npy")
count5all,countall,sumdata=0,0,0
global_point = 0
batch_size = 1
start = time()
for i in range(probe_data.__len__()):
    data,label_temp = load_data_cross(batch_size = batch_size,data_source = probe_data)#
    output = model(data)#1,512
    output = output.squeeze()
    output = output.detach().cpu().numpy()
    top1,top5 = calcul(output,result,label)
    #print(label_temp,top1,top5)
    if batch_size==1:
        label_temp = int(label_temp[0])
    if label_temp == top1:
        countall=countall+1
    if label_temp in top5:
        count5all = count5all+1
    sumdata = i
    if(i%500==0 and i>0):
        print('test_num: {} [{}/{} ({:.0f}%)] top1:{:.2f} top5:{:.2f} time:{:.2f}'.format(
        i, i, probe_data.__len__(),100. * i / probe_data.__len__(), countall/sumdata,count5all/sumdata),time()-start)
#def calcul_gallery(probe,):

'''

batch_size = 50
countall = 0
count5all = 0
totalnum = 0
for ita in range(50):

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
    totalnum = totalnum + batch_size
    #print("top1:",countall/batch_size, " top5:",count5all/batch_size)

print("top1:",countall/totalnum, " top5:",count5all/totalnum)

'''
#test
