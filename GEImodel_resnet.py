
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


#from datagen inport MyDataset
#torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
plt.ion()   # 交互模式
use_cuda = torch.cuda.is_available()

imgpath = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/GEI/'
#imgpath = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/GEI/'
root = '/home/xiaoqian/GeiGait/'
# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path)
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
        #img = self.loader(imgpath + fn)
        img = self.loader(fn)
        #img = img.resize((224,224))
        imgdata = np.array(img)
        #print(imgdata.shape)
        imgdata = imgdata[np.newaxis,:]
        return imgdata,label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=root+'datalist/casia_stn_train.txt', transform=transforms.ToTensor())
#print(img1[0].shape,torch.tensor(img1[0]))
test_data=MyDataset(txt=root+'datalist/casia_stn_test.txt', transform=transforms.ToTensor())


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
        #super
        self.conv1 = nn.Conv2d(1, 64, kernel_size=17, stride=1, padding=[0,20],bias=False)#128*128
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

    def stn(self, x):
        xs = self.localization(x)

        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size([x.size(0),1,128,88]))
        x = F.grid_sample(x, grid)
        x = torch.ceil(torch.div(x,255))
        x = torch.mul(x,255)

        return x
    def Gei(self,x):
        x = torch.transpose(x,0,1)
        #print('dim: ',int(x.size(1)/20))
        groups = int(x.size(1)/20)
        weight = torch.ones([groups,20,1,1]).cuda()
        weight = torch.div(weight,int(x.size(1))/20)
        fusion = F.conv2d(x, weight,stride=1,groups=groups).cuda()
        return torch.transpose(fusion,0,1)

    def forward(self, x):
        #x = self.stn(x)
        #return x
        #x = self.Gei(x) 
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
def load_test_cross(batch_size=10):
    global global_point
    #print('globale_point: ',global_point)
    data=[]
    label = []
    for batchpt in range(batch_size):
        #data.append(train_data.__get.item__(global_point+batchpt)[0])
        #label.append(train_data.__getitem__(global_point+batchpt)[2])
        tempdata,templabel = test_data.__getitem__(global_point)

        data.append(tempdata)
        label.append(templabel)
        global_point = global_point + 1
        #print('batchpt',batchpt,'global_point:',global_point)GeiNet30000.pkl
    #global_point = global_point + batch_size
    return torch.from_numpy(np.array(data)).float(),torch.from_numpy(np.array(label)).long()


#
def train(ita_num):
    countall = 0
    count5all = 0
    batch_size = 100
    for ita in range(ita_num):

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
            countall=0
            count5all=0
        if(ita>=5000 and ita%2000==0):
            torch.save(model,'./model/stn_casia'+str(ita)+'.pkl')



def test():
    model.eval()
    
    batch_size = 50
    countall = 0
    count5all = 0
    totalnum = 0
    for ita in range(50):

        #print("global_point:",global_point)
        data1,label1 = load_test_cross(batch_size=batch_size) #5
        data = torch.autograd.Variable(data1.cuda())
        '''
        testsample = data1.squeeze()
        testsample = testsample.view(1,-1).squeeze()
        print(testsample.shape)
        testsample = testsample.detach().cpu().numpy().tolist()
        print(set(testsample))
        return 
        '''
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
        print("top1:",countall/totalnum, " top5:",count5all/totalnum)

    print("top1:",countall/totalnum, " top5:",count5all/totalnum)




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
    batch_size = 30
    while(batch_idx < len(train_loader.dataset)/batch_size):
        batch_idx = batch_idx + 1
        data,aim = load_data_cross(batch_size=batch_size,data_source = test_data)
        data = Variable(data, volatile=True)

        
        if use_cuda:
            data = data.cuda()
        input_tensor = data.cpu().data[:30]
        transformed_input_tensor = model.stn(data).cpu().data[:30]
        fusion_input_tensor = model.Gei(model.stn(data)).cpu().data[:4]
        #print(transformed_input_tensor.size())
        '''
        result_img =transformed_input_tensor[0][0].numpy()
        #print(set(flatten(result_img.tolist())))
        print(result_img[result_img>2])
        plt.imshow(result_img)
        plt.show()
        plt.savefig('model/result_img')
        '''
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
'''
model = Net(BasicBlock,[2,2,2,2],num_classes=150)
if use_cuda:
    model.cuda()
pretrained_file = 'model/STN_gei250.pkl'
pre_model = torch.load(pretrained_file)
pre_model = pre_model.state_dict()
model_dict = model.state_dict()
weight = {k:v for k,v in pre_model.items() if k in model_dict}
#print(weight.keys())
model_dict.update(weight)
model.load_state_dict(model_dict)
print(list(model.state_dict().keys()))
for i,p in enumerate(model.parameters()):
    if i<10:
        p.requires_grad = False
    else:
        p.requires_grad = True

#print(list(model.state_dict().keys())[10:], len(model.state_dict().keys()))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=0.001,momentum=0.9)
#model.eval()
writer = SummaryWriter(comment="stn_casia")
start = time()
#
train(10000+1)
    #test()

# 在一些输入批次中可视化空间转换网络 (STN) 的转换
writer.export_scalars_to_json("./log/stn_casia.json")
writer.close()

#testm()
'''
pretrained_file = 'model/stn_casia10000.pkl'
model = torch.load(pretrained_file)
model.eval()
test()
#testm()
#visualize_stn(2)
#plt.ioff()

