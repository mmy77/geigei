from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tensorboardX import SummaryWriter
from time  import time
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
#__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']
#from datagen inport MyDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
plt.ion()   # 交互模式


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
            imgs.append((words[0],words[1],words[2]))  
        self.imgs = imgs
        self.transform = transforms.Compose([
            transforms.CenterCrop((300,300)),
            transforms.Grayscale(num_output_channels=1),
            #transforms.ToTensor()
            ]
            )
        
        self.target_transform = transforms.Compose([
            #transforms.CenterCrop((224,224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5),std=(0.5,0.5))
            ])
        self.loader = loader

    def __getitem__(self, index):

        global global_point 
        global epoch

        if(index >= self.__len__()):
            epoch = epoch+1
        index = int(index%(self.__len__()))

        length = len('/mnt/ftp/datasets/dataset-gait/CASIA/DatasetB/silhouettes/113/cl-01/018/')
        data_num, input_data, target_data = self.imgs[index]
        #dirname = input_data[::-1][input_data[::-1].find('/'):][::-1]#../../
        #print(fn)
        train_img = self.loader(input_data) 
        #target_img = self.loader(target_data)  
        train_id = int(target_data)
       # train_img = np.expand_dims(np.array(train_img),axis=0)
        #target_img = np.expand_dims(np.array(target_img),axis=0)
        #print(train_img.shape)
        train_img = self.transform(train_img)
        train_img = np.array(train_img)
        #target_img = self.target_transform(target_img)
        train_dir =  input_data[:length]
        #print(input_data,target_data)
        global_point = global_point + 1
        return train_img, train_id, data_num
        
        #return fn, label

    def __len__(self):
        return len(self.imgs)
trainroot = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/silhouettes/'

train_data=MyDataset(txt=root+'datalist/train_sgei.txt', transform=transforms.ToTensor())
#print(img1[0].shape,torch.tensor(img1[0]))
test_data=MyDataset(txt=root+'datalist/test_sgei.txt', transform=transforms.ToTensor())

#trainset = MyDataset(path_file=pathfile,list_file=trainlist,numJoints=6,type=False)
train_loader = torch.utils.data.DataLoader(train_data,batch_size = 30, shuffle=False)
#testset = MyDataset(path_file=pathFile,list_file=testList,numJoints=6,type=False)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=30,shuffle=False)
'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GEInet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        


        self.inplanes = 64
        super(GEInet, self).__init__()

        #localization
        self.localization = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=4,stride=2),#149,149
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=4,stride=2),#36,36
            nn.MaxPool2d(2, stride=2),#18,18
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=4,stride=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # 3 * 2 仿射矩阵 (affine matrix) 的回归器
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),#10,4,4
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 用身份转换 (identity transformation) 初始化权重 (weights) / 偏置 (bias)
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        for i in self.parameters():
            i.requires_grad=False
        #resnetresnet
        #self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)#128*128
        self.conv1 = nn.Conv2d(1, 64, kernel_size=17, stride=1, padding=[0,20],bias=False)#128*128
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=[1,2],bias=False)#122x82

        #output 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#56*56
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#56*56
        #self.maxpool = nn.MaxPool2d(kernel_size=[27,6], stride=1)
        #output 56x56
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#28*28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#14*14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#7*7
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def stn(self, x):
        xs = self.localization(x)
        #print(xs.size())

        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size([x.size(0),1,128,88]))
        x = F.grid_sample(x, grid)

        return x

    def Gei(self,x):#200,1,128,88   #10,20,1,128,88
        result = x
        #print('x,size / 20',x.size(0),x.size(0)/20)
        for j in range(int(x.size(0)/20)):#1,200,128,88
            temp = x[j*20:j*20+20,:]
            temp = torch.transpose(temp,0,1)#1,20-40,128,88
            fusion = nn.Conv2d(temp.size(1),temp.size(1),kernel_size=1,stride=1).cuda()
            fusion.weight.data.fill_(1/x.size(1))
            for i in fusion.parameters():
                i.requires_grad = False
            temp = fusion(temp)
            if j==0:

                result = temp
            else:
                result = torch.cat((result,temp),1)
        return result


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stn(x)
        #xs = x
        x = self.Gei(x)
        x = torch.transpose(x,0,1)#30,1,128,88
        #x = x.expand(30,3,224,224)
        #print(x.size())
        x = self.conv1(x)
        #print("conv",x.shape)
        x = self.bn1(x)
       # print("bn",x.shape)
        x = self.relu(x)
        x = self.maxpool(x)
       # print("maxp",x.shape)
        x = self.layer1(x)
       # print("maxp",x.shape)
        x = self.layer2(x)
       # print("maxp",x.shape)
        x = self.layer3(x)
       # print("maxp",x.shape)
        x = self.layer4(x)
       # print("maxp",x.shape)

        x = self.avgpool(x)
        #print('poll:',x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def geinet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GEInet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def load_data_cross(batch_size=200):
    global global_point
    #print('globale_point: ',global_point)
    data=[]
    label = []
    count = 0
    block_count = 0
    tempblock = []
    tempblock2 = []
    tempdata,templabel,data_id = train_data.__getitem__(global_point)
    pre_id = data_id
    tempblock.append(tempdata)
    tempblock2.append(templabel)
    count = count + 1
    while(1):
        tempdata,templabel,data_id = train_data.__getitem__(global_point)
        if(data_id>pre_id and count<20):#not enough img
            tempblock.append(tempdata)
            tempblock2.append(templabel)
            count = count + 1
        if(data_id > pre_id and count == 20):# up to 20 img
            data = data + tempblock
            label = label + tempblock2
            block_count = block_count + 1
            tempblock = []
            tempblock2 = []
            count = 0
            if(block_count >= batch_size/20):
                break
        if(count<20 and data_id<pre_id):#another file
            tempblock=[]
            tempblock2 = []
            count = 0
            pre_id = data_id
    #print(np.array(data).shape(),np.array(label).shape())
    data = np.array(data)
    label = np.array(label)
    data = data[:,np.newaxis,:,:]
    return torch.from_numpy(data).float(),torch.from_numpy(label).long()
def train(epoch):
    model.train()
    #criterion = nn.MSELoss(reduce=True,size_average=True)
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target,train_id) in enumerate(train_loader):
        #print(i)
        print(data.size(),target.size(),train_id.size())
        if use_cuda:
            data, target = data.cuda(), target.cuda()
            train_id = train_id.cuda().long()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #loss = criterion(output, target)
        loss = criterion(output,train_id)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            end = time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]),'time:',end-start)
            writer.add_scalar('loss/x',loss,batch_idx)
    if epoch%50==0:
        torch.save(model,'./model/Stn_Gei'+str(epoch)+'.pkl')

def train_all(epoch):
    print("epoch at train begin: ", epoch)
    pre_epoch = epoch
    model.train()
    #criterion = nn.MSELoss(reduce=True,size_average=True)
    criterion = nn.CrossEntropyLoss()
    countall = 0
    count5all = 0
    all_len = 0
    batch_idx = 0
    while(1):
        #print(i)
        print("batch_idx:",batch_idx)
        data, target = load_data_cross(batch_size = 200)
        if(epoch > pre_epoch):
            break
        #print(data.size(),target.size())
        if use_cuda:
            data, target = data.cuda(), target.cuda()
            #train_id = train_id.cuda().long()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target.cuda())
        loss.backward()
        optimizer.step()

        tout = output.cpu().detach().numpy()
        pred = []
        pred5 = []
        maxp = -1
        max_index = 0
        for i in range(len(data)):
            one_out = tout[i,:]
            max_index = np.argsort(-one_out)[0]
            max5_index = np.argsort(-one_out)[0:5]
            pred.append(max_index)
            pred5.append(max5_index) #
        pred = np.array(pred)
        label2=target.cpu().detach().numpy() #answer
        static = pred-label2
        zero = static==0
        zero_num = static[zero]
        countall = countall + zero_num.size
        for i in range(len(pred5)):
            if label2[i] in pred5[i]:
                count5all = count5all + 1
        #print(out.shape,label1.shape)       
        all_len = all_len + len(data)

        #loss = criterion(output, target)


        batch_idx = batch_idx + 1
        if batch_idx % 20 == 0:
            end = time()
            accuracy = countall / all_len
            top5acc  = count5all / all_len
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc5: [{}/{} {:.6f}]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], count5all,all_len,accuracy),'time:',\
                end-start,"accuracy: "+ str(accuracy))
            countall = 0
            count5all = 0
            all_len = 0
            writer.add_scalar('loss/x',loss,batch_idx)
            writer.add_scalar('accuracy/x',accuracy,batch_idx)
            writer.add_scalar('top5/x',top5acc,batch_idx)
    
    if epoch%20==0:
        torch.save(model,'./model/Stn_Gei'+str(epoch)+'.pkl')





def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def testm():
    model = GEInet(BasicBlock,[2,2,2,2],num_classes=150).cuda()
    pretrained_file = 'model/StnGei200.pkl'
    model = torch.load(pretrained_file)
    model.eval()
def visualize_stn(num):
    # 得到一批输入数据
    #data, _ = next(iter(test_loader))
    #data, aim = iter(test_loader).next()
    order = 0
    #for data,aim in test_loader:
    for i in range(num):
        data,aim = load_data_cross()
        print(order,data.size(),aim.size())
        data = Variable(data, volatile=True)

        aim = Variable(aim)
        
        if use_cuda:
            data = data.cuda()

       # aim_tensor = aim.data
            
        input_tensor = data.cpu().data[:20]
        transformed_input_tensor = model.stn(data).cpu().data[:20]
        fusion_input_tensor = torch.transpose(model.Gei(model.stn(data)),0,1).cpu().data[:20]
        #print(transformed_input_tensor.size())

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor,padding = 2))

        #aim_grid = convert_image_np(
         #   torchvision.utils.make_grid(aim_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        fusion_grid = convert_image_np(
            torchvision.utils.make_grid(fusion_input_tensor))
        # 并行地 (side-by-side) 画出结果
        f, axarr = plt.subplots(2, 2)
        axarr[0,0].imshow(in_grid)
        axarr[0,0].set_title('Dataset Images')

        axarr[0,1].imshow(out_grid)
        axarr[0,1].set_title('Transformed Images')

        #axarr[1,0].imshow(aim_grid)
        #axarr[1,0].set_title('groundtruth ')

        axarr[1,1].imshow(fusion_grid)
        axarr[1,1].set_title('output_fusion')

        plt.savefig('model/'+str(order)+'stn.png')
        plt.show()
        plt.pause(1)
        plt.close()
        order = order+1
        if(order>5):
            break
    #break


use_cuda = torch.cuda.is_available()
model = GEInet(BasicBlock,[2,2,2,2],num_classes=150).cuda()
if use_cuda:
    model.cuda()
'''
optimizer = optim.SGD(model.parameters(), lr=0.01)

writer = SummaryWriter()
start = time()

for epoch in range(1, 200 + 1):
    train(epoch)
    #test()

# 在一些输入批次中可视化空间转换网络 (STN) 的转换
writer.export_scalars_to_json("./log/geimodel.json")
writer.close()
'''
#testm()
global global_point
global_point = 0
global  epoch
epoch = 1
pretrained_file = 'model/StnGei200.pkl'
model = torch.load(pretrained_file)
#model.eval()
optimizer = optim.SGD(model.parameters(), lr=0.01)

writer = SummaryWriter()
start = time()

#for epoch in range(1, 200 + 1):
#    train_all(epoch)
    #test()
while(epoch < 20):
    train_all(epoch)

# 在一些输入批次中可视化空间转换网络 (STN) 的转换
writer.export_scalars_to_json("./log/geimodel.json")
writer.close()
visualize_stn(5)
#plt.ioff()
