
from __future__ import print_function
import torch
import torch.nn as nn
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
import cv2

#from datagen inport MyDataset
#torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
plt.ion()   # 交互模式
use_cuda = torch.cuda.is_available()

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
            ])
        
        self.target_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ])
        self.loader = loader

    def __getitem__(self, index):

        global global_point 
        global_point = global_point+1
        index = int(index%(self.__len__()))
        order, input_data, target_data = self.imgs[index]
        train_img = self.loader(input_data) 
        train_img = self.transform(train_img)
        train_img = np.array(train_img,np.float32)[np.newaxis,:]
        train_img = train_img/255
        target_img = int(target_data)
        imgname = input_data.split('/')[-1]
        return int(order),train_img,target_img,imgname

    def __len__(self):
        return len(self.imgs)
trainroot = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/silhouettes/'

train_data=MyDataset(txt=root+'datalist/train_sgei.txt')
#print(img1[0].shape,torch.tensor(img1[0]))
test_data=MyDataset(txt=root+'datalist/test_sgei.txt')

train_loader = torch.utils.data.DataLoader(train_data,batch_size=20, shuffle=False, num_workers=8,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=20,shuffle=False,num_workers=8,drop_last=True)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
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
class Net(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(Net, self).__init__()

        # 3 * 2 仿射矩阵 (affine matrix) 的回归器



        #super
        self.conv1 = nn.Conv2d(1, 64, kernel_size=17, stride=1, padding=[0,20],bias=False)#128*128
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#56*56
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#28*28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#14*14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#7*7
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_m = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

    def get_grid(self,x):
        rect = []
        for i in range(x.size(0)):
            #print("x:",x)
            rect_one = []
            one_img = x[i]
            one_img = one_img.squeeze()
            one_img = one_img.cpu().numpy().astype(np.uint8)
            #print(one_img.shape,one_img)
            x1,y,w,h = cv2.boundingRect(one_img)#x,y,w,h
            new_w = 88.0*h/128
            rect_one.append(int(x1-new_w/2+w/2))
            rect_one.append(y)
            rect_one.append(int(x1+new_w/2+w/2))
            rect_one.append(y+h)

            #print(rect_one)
            rect.append(rect_one)
        return rect
    def rect_crop(self,x,rect):# if rect<0
        new_tensor = torch.empty(x.size(0),1,128,88).cuda()
        x = torch.nn.functional.pad(x,(40,40,40,40),value=0,mode='constant')
        for i in range(x.size(0)):
            #print(x[i].shape)
            #print(rect[i])

            crop_area = x[i][:,rect[i][1]+40:rect[i][3]+41,rect[i][0]+40:rect[i][2]+41]
            crop_area = crop_area.unsqueeze(0)
            #print(crop_area.shape)
            #print(crop_area)

            new_tensor[i] = nn.functional.upsample(crop_area,size=(128,88),mode='nearest')#,mode=bilinear,align_corners=True)
        #print(new_tensor.shape)
        return new_tensor
    def stn(self, x):#batch,1,300,300
        rect = self.get_grid(x)
        #print(len(rect),x.shape)
        x = self.rect_crop(x,rect)
        #x = nn.Upsample(size=(128,88),align_corners=True)
        return x
    def Gei(self,x):
        x = torch.transpose(x,0,1)
        groups = int(x.size(1)/20)#
        weight = torch.ones([groups,20,1,1]).cuda()
        weight = torch.div(weight,20)
        fusion = F.conv2d(x, weight,stride=1,groups=groups).cuda()
        return torch.transpose(fusion,0,1)
    def forward(self, x):
        x = self.stn(x)
        #return x
        x = self.Gei(x) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_m(x)
        return x
def load_data_cross(batch_size=200,data_source=train_data):
    global global_point
    data=[]
    label = []
    count = 0
    block_count = 0
    tempblock = []
    tempblock2 = []
    imgname = []
    data_id, tempdata,templabel,name=data_source.__getitem__(global_point)
    pre_id = data_id
    tempblock.append(tempdata)
    tempblock2.append(templabel)
    count = count + 1
    while(1):
        data_id, tempdata,templabel,name = data_source.__getitem__(global_point)
        if(data_id>pre_id and count<20):#not enough img
            tempblock.append(tempdata)
            tempblock2.append(templabel)
            count = count + 1
        if(data_id > pre_id and count == 20):# up to 20 img
            data = data + tempblock
            label.append(tempblock2[0])
            imgname.append(name)
            block_count = block_count + 1
            tempblock = []
            tempblock2 = []
            count = 0
            if(block_count >= batch_size/20):
                break
        if(count<20 and data_id<=pre_id):#another file
            tempblock=[]
            tempblock2 = []
            tempblock.append(tempdata)
            tempblock2.append(templabel)
            count = 1
            pre_id = data_id

    data = np.array(data)
    label = np.array(label)#int
    return torch.from_numpy(data).float(),torch.from_numpy(label).long()#,imgname

def train(epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    batch_size,batch_idx,countall,count5all,sum_static =1000,0,0,0,0
    ita_num = len(train_loader.dataset)/batch_size
    while(batch_idx < ita_num):
        batch_idx = batch_idx + 1
        data,target = load_data_cross(batch_size = batch_size,data_source = train_data)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        out = output.cpu().detach().numpy()
        #print(out.shape)
        pred,pred5 = [],[]
        maxp,max_index= -1,0
        for i in range(out.shape[0]):
            one_out = out[i,:]
            max_index = np.argsort(-one_out)[0]
            max5_index = np.argsort(-one_out)[0:5]
            pred.append(max_index)
            pred5.append(max5_index) #
        pred = np.array(pred)
        aim = np.array(target)
        static = pred-aim
        countall = countall + static[static==0].size
        for i in range(len(pred5)):
            if aim[i] in pred5[i]:
                count5all=count5all+1
        sum_static = sum_static + out.shape[0]
        if batch_idx % 2 == 0:
            return
            end = time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / ita_num, loss.data[0]),'time:',end-start)
            print('total_seq:',sum_static, " top1:",countall,countall/sum_static,\
                " top5:",count5all/sum_static)
            #print("top1: ",top1," top5:",top5)
            writer.add_scalar('loss/x',loss,batch_idx)
            writer.add_scalar('top5/x',count5all/sum_static,batch_idx)
            writer.add_scalar('top1/x',countall/sum_static,batch_idx)
            sum_static,countall,count5all = 0,0,0 
    if epoch%1==0:
        torch.save(model,'./model/stn_casia_train'+str(epoch)+'.pkl')

def saveGei():
    
    rootsavep = '/home/xiaoqian/GeiGait/data_aug/train/'
    model.eval()
    batch_size = 20
    batch_idx = 0
    global global_point
    start = time()
    count_num = 0
    '''
    while(global_point<train_data.__len__()):
        batch_idx = batch_idx + 1
        data,aim,imgname = load_data_cross(batch_size=batch_size,data_source = train_data)
        data = Variable(data, volatile=True)
        if use_cuda:
            data = data.cuda()
        out = model.Gei(model.stn(data))
        #out = out.squeeze()
        for i in range(len(imgname)):
            #print(out[i].shape())
            imgdata = out[i].squeeze()
            imgdata = imgdata.cpu().detach().numpy()
            img = Image.fromarray(imgdata)
            img = img.convert('L')
            savepath = rootsavep + imgname[i]
            print(batch_idx, savepath,time()-start)
            img.save(savepath)
        if(global_point>10):
            global_point = global_point-10 #more gei every seq
    '''
    rootsavep = '/home/xiaoqian/GeiGait/data_aug/test/'
    batch_idx=0
    global_point = 0
    while(global_point<test_data.__len__()):
        batch_idx = batch_idx + 1
        data,aim,imgname = load_data_cross(batch_size=batch_size,data_source = test_data)
        data = Variable(data, volatile=True)
        if use_cuda:
            data = data.cuda()
        out = model.Gei(model.stn(data))
        #out = out.squeeze()
        for i in range(len(imgname)):
            #print(out[i].shape())
            imgdata = out[i].squeeze()
            imgdata = imgdata.cpu().detach().numpy()
            img = Image.fromarray(imgdata)
            img = img.convert('L')
            savepath = rootsavep + imgname[i]
            print(batch_idx,savepath,time()-start)
            img.save(savepath)
        if(global_point>10):
            global_point = global_point-10 #more gei every seq
def test():
    model.eval()
    batch_size,batch_idx,countall,count5all,sum_static = 200,0,0,0,0
    start = time()
    while(batch_idx < len(test_loader.dataset)/batch_size):
        batch_idx = batch_idx + 1
        data,aim,imgname = load_data_cross(batch_size=batch_size,data_source = test_data)
        data = Variable(data, volatile=True)
        if use_cuda:
            data = data.cuda()
        out = model(data)
        out = out.cpu().detach().numpy()
        #print(out.shape)
        pred,pred5 = [],[]
        maxp,max_index= -1,0
        for i in range(out.shape[0]):
            one_out = out[i,:]
            max_index = np.argsort(-one_out)[0]
            max5_index = np.argsort(-one_out)[0:5]
            pred.append(max_index)
            pred5.append(max5_index) #
        pred = np.array(pred)
        aim = np.array(aim)
        static = pred-aim
        countall = countall + static[static==0].size
        for i in range(len(pred5)):
            if aim[i] in pred5[i]:
                count5all=count5all+1
        sum_static = sum_static + out.shape[0]
        if(batch_idx%10==0):
            end = time()
            print('batch_idx:',batch_idx,sum_static, " top1:",countall,countall/sum_static,\
                " top5:",count5all/sum_static,'time:',end-start)
            sum_static,countall,count5all = 0,0,0

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
def visualize_stn(num):
    order = 0
    batch_idx = 0
    batch_size = 40
    while(batch_idx < len(train_loader.dataset)/batch_size):
        batch_idx = batch_idx + 1
        data,aim = load_data_cross(batch_size=batch_size,data_source = test_data)
        data = Variable(data, volatile=True)
        if use_cuda:
            data = data.cuda()
        input_tensor = data.cpu().data[:30]
        transformed_input_tensor = model.stn(data).cpu().data[:30]
        fusion_input_tensor = model.Gei(model.stn(data)).cpu().data[:4]
        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor,padding = 2))

        #aim_grid = convert_image_np(
        #    torchvision.utils.make_grid(aim_tensor))
        
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
        
        if(order>=num):
            break

global global_point
global_point = 0

model = Net(BasicBlock,[2,2,2,2],num_classes=150)
if use_cuda:
    model.cuda()
#print(list(model.state_dict().keys()))

pretrained_file = 'model/casia_stnLT3000.pkl'
pre_model = torch.load(pretrained_file)

pre_model = pre_model.state_dict()
model_dict = model.state_dict()
weight = {k:v for k,v in pre_model.items() if k in model_dict}
print(weight.keys())
model_dict.update(weight)
model.load_state_dict(model_dict)

for i,p in enumerate(model.parameters()):
        p.requires_grad = True
#print(list(model.state_dict().keys())[10:], len(model.state_dict().keys()))
optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=0.0001,momentum=0.9)
#model.eval()
writer = SummaryWriter(comment='crop_train')
start = time()
#
for epoch in range(1, 1 + 1):
    train(epoch)
# 在一些输入批次中可视化空间转换网络 (STN) 的转换
writer.export_scalars_to_json("./log/crop_train.json")
writer.close()

#testm()

#pretrained_file = 'model/GeiNet_resnet10000.pkl'
#model = torch.load(pretrained_file)
model.eval()
#saveGei()
#test()
visualize_stn(2)
#plt.ioff()

