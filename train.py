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
import argparse
parser = argparse.ArgumentParser(description='train mode')
parser.add_argument('--data_size',type=str,default='stnLT')
parser.add_argument('--batch_size',type=int,default=20)
parser.add_argument('--ita_times',type=int,default=3000)
parser.add_argument('--num_classes',type=int,default=150)
parser.add_argument('--imgpath',type=str,default='/home/lxq/GeiGait/data/')
args = parser.parse_args()
data_size = args.data_size
batch_size = args.batch_size
ita_times = args.ita_times
num_classes = args.num_classes
if(data_size=='ST'):
    num_classes = 25
elif(data_size=='MT'):
    num_classes = 63
else:
    num_classes=75
imgpath = args.imgpath
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
        imgdata = imgdata.swapaxes(0,2)
        return imgdata,label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=root+'datalist/'+data_size+'_train_list.txt', transform=transforms.ToTensor())
test_data=MyDataset(txt=root+'datalist/test_casia.txt', transform=transforms.ToTensor())


def load_data_cross(batch_size=10):
    global global_point
    data=[]
    label = []
    for batchpt in range(batch_size):
        tempdata,templabel = train_data.__getitem__(global_point)
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

pretrained_file = 'resnet18-5c106cde.pth'
model = ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes).cuda()
model = load_model(model,pretrained_file)
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
global_point = 0
print('model ok')

start = time()
countall,count5all,batch_size = 0,0,20
'''
with SummaryWriter(comment='casia_MT') as w:
    w.add_graph(model,data1)
'''
writer = SummaryWriter(comment='casia_'+data_size)


for ita in range(ita_times+1):

    data1,label1 = load_data_cross(batch_size=batch_size) #5
    criterion = nn.CrossEntropyLoss()
    data = torch.autograd.Variable(data1.cuda())
    optimizer.zero_grad()
    out = model(data)
    #print(out.shape)
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
        countall=0
        count5all=0
    if(ita>2000 and ita%1000==0):
        torch.save(model,'./model/casia_'+data_size+str(ita)+'.pkl')

    #print(out.size(),out)
writer.export_scalars_to_json('./log/casia_'+data_size+'.json')
writer.close()

#test
