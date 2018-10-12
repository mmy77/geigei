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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def get_optim_policies(model=None,modality='RGB',enable_pbn=True):
    #print('this is not used/!#################################')
    '''''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn2, and many all bn3.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model==None:
        log.l.info('no model!')
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])
              
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m,torch.nn.BatchNorm2d):
            bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    slow_rate=0.7
    n_fore=int(len(normal_weight)*slow_rate)
    slow_feat=normal_weight[:n_fore] # finetune slowly.
    slow_bias=normal_bias[:n_fore] 
    normal_feat=normal_weight[n_fore:]
    normal_bias=normal_bias[n_fore:]

    return [
        {'params': first_conv_weight, 'lr_mult': 1 , 'decay_mult': 1, 'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 1, 'decay_mult': 0, 'name': "first_conv_bias"},
        {'params': slow_feat, 'lr_mult': 10, 'decay_mult': 1, 'name': "slow_feat"},
        {'params': slow_bias, 'lr_mult': 10, 'decay_mult': 0,'name': "slow_bias"},
        {'params': normal_feat,'decay_mult': 1,'lr':0.01,'name': "normal_feat"},
        {'params': normal_bias,  'decay_mult':1, 'lr':0.01,'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'name': "BN scale/shift"},
        #{'params':model.myfc.parameters(), 'lr': 0.001},
    ]

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
        #print("swap:",imgdata.shape, imgdata.max())
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
        #data.append(train_data.__get.item__(global_point+batchpt)[0])
        #label.append(train_data.__getitem__(global_point+batchpt)[2])
        tempdata,templabel = train_data.__getitem__(global_point)

        data.append(tempdata)
        label.append(templabel)
        global_point = global_point + 1
        #print('batchpt',batchpt,'global_point:',global_point)GeiNet30000.pkl
    #global_point = global_point + batch_size
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

pretrained_file = 'resnet18-5c106cde.pth'
#pretrained_file = 'model/GeiNet30000.pkl'
model = ResNet(BasicBlock,[2,2,2,2],num_classes=150).cuda()
model = load_model(model,pretrained_file)
num_classes = 150

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(),lr=0.00001,momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
#optimizer = optim.SGD(get_optim_policies(model=model),lr=0.0001,momentum=0.9)
#print(model)
global_point = 0
print('model ok')
model = model.cuda()
logfile = './log/GeiNet_tum.txt'
f = open(logfile,'w')
start = time()
countall = 0
count5all = 0
batch_size = 20
writer = SummaryWriter()


for ita in range(2001):

    data1,label1 = load_data_cross(batch_size=batch_size) #5

    data = torch.autograd.Variable(data1.cuda())
    optimizer.zero_grad()
    out = model(data)
    tout = out.cpu().detach().numpy()
    #print(out.shape,label2.shape)
    pred = []
    pred5 = []
    maxp = -1
    max_index = 0
    for i in range(batch_size):
        one_out = tout[i,:]
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
    for i in range(len(pred5)):
        if label2[i] in pred5[i]:
            count5all = count5all + 1


    loss = criterion(out,label1.cuda())
    loss.backward()
    optimizer.step()
    end = time()
    print("ita:",ita," loss:",str(loss.cpu().detach().numpy())," time: ",str(int(end-start)), countall, count5all)
    if(ita%50==0):
        
        accuracy = countall/(50.0*batch_size)
        top5acc = count5all/(50.0*batch_size)
        writer.add_scalar('loss/x',loss,ita)
        writer.add_scalar('accuracy/x',accuracy,ita)
        writer.add_scalar('top5/x',top5acc,ita)
        print("ita:",ita," loss:",str(loss.cpu().detach().numpy()), " time: ",str(int(end-start)))
        print("accuracy: "+ str(accuracy), " top5: " + str(top5acc))
        
        f.write(str(ita)+' ' + str(loss.cpu().detach().numpy())+' '+str(accuracy)+'\n')
        countall=0
        count5all=0
    if(ita>=300 and ita%100==0):
        torch.save(model,'./model/GeiNet_tum'+str(ita)+'.pkl')

    #print(out.size(),out)
writer.export_scalars_to_json("./log/GeiNet_tum.json")
writer.close()



#test
