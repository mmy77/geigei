
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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
            imgs.append((words[0],words[1]))  
        self.imgs = imgs
        self.transform = transforms.Compose([
            transforms.CenterCrop((300,300)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
            ]
            )
        
        self.target_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5),std=(0.5,0.5))
            ])
        self.loader = loader

    def __getitem__(self, index):

        global global_point 
        index = int(index%(self.__len__()))

        input_data, target_data = self.imgs[index]
        #print(fn)
        train_img = self.loader(input_data) 
        target_img = self.loader(target_data)  

       # train_img = np.expand_dims(np.array(train_img),axis=0)
        #target_img = np.expand_dims(np.array(target_img),axis=0)
        #print(train_img.shape)
        train_img = self.transform(train_img)
        target_img = self.target_transform(target_img)
        #print(input_data,target_data)
        return train_img, target_img
        
        #return fn, label

    def __len__(self):
        return len(self.imgs)
trainroot = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/silhouettes/'

train_data=MyDataset(txt=root+'datalist/train_sample.txt', transform=transforms.ToTensor())
#print(img1[0].shape,torch.tensor(img1[0]))
test_data=MyDataset(txt=root+'datalist/test_sample.txt', transform=transforms.ToTensor())

#trainset = MyDataset(path_file=pathfile,list_file=trainlist,numJoints=6,type=False)
train_loader = torch.utils.data.DataLoader(train_data,batch_size = 30, shuffle=True, num_workers=8)
#testset = MyDataset(path_file=pathFile,list_file=testList,numJoints=6,type=False)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=30,shuffle=False,num_workers=8)
'''
# 训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        '''
        # 空间转换本地网络 (Spatial transformer localization-network)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=10,stride=8),#64,48
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=10),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        '''
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

    # 空间转换网络的前向函数 (Spatial transformer network forward function)
    def stn(self, x):
        xs = self.localization(x)
        #print(xs.size())

        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size([x.size(0),1,128,88]))
        x = F.grid_sample(x, grid)

        return x

    def Gei(self,x):
        x = torch.transpose(x,0,1)

        fusion = nn.Conv2d(x.size(1),1,kernel_size=1,stride=1).cuda()
        fusion.weight.data.fill_(1/x.size(1))
        for i in fusion.parameters():
            i.requires_grad = False
        return fusion(x)
    def forward(self, x):
        # 转换输入
        x = self.stn(x)
        #print(x.size()) #30,1,128,88
        x_result = self.Gei(x)
        return x
        # 执行常规的正向传递
        '''
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''



def train(epoch):
    model.train()
    criterion = nn.MSELoss(reduce=True,size_average=True)
    #print(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
       # print(data.size(),target.size())
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            end = time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]),'time:',end-start)
            writer.add_scalar('loss/x',loss,batch_idx)
    if epoch%50==0:
        torch.save(model,'./model/STN_gei'+str(epoch)+'.pkl')
#
# 一个简单的测试程序来测量空间转换网络 (STN) 在 MNIST 上的表现.
#


def test():
    model.eval()
    
    test_loss = 0
    correct = 0
    criterion = nn.MSELoss(reduce=True,size_average=True)
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # 累加批loss
        test_loss += criterion(output, target, size_average=False).data[0]
        # 得到最大对数几率 (log-probability) 的索引.
        #pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
    #      .format(test_loss, correct, len(test_loader.dataset),
    #              100. * correct / len(test_loader.dataset)))

def testm():
    model = Net()
    pretrained_file = 'model/STN_gei200.pkl'
    model = torch.load(pretrained_file)
    model.eval()


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# 我们想要在训练之后可视化空间转换层 (spatial transformers layer) 的输出, 我们
# 用 STN 可视化一批输入图像和相对于的转换后的数据.


def visualize_stn(num):
    # 得到一批输入数据
    #data, _ = next(iter(test_loader))
    #data, aim = iter(test_loader).next()
    order = 0
    for data,aim in test_loader:
        print(order)
        data = Variable(data, volatile=True)

        aim = Variable(aim)
        
        if use_cuda:
            data = data.cuda()

        aim_tensor = aim.data
            
        input_tensor = data.cpu().data
        transformed_input_tensor = model.stn(data).cpu().data
        fusion_input_tensor = model.Gei(model.stn(data)).cpu().data
        #print(transformed_input_tensor.size())

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor,padding = 2))

        aim_grid = convert_image_np(
            torchvision.utils.make_grid(aim_tensor))

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

        axarr[1,0].imshow(aim_grid)
        axarr[1,0].set_title('groundtruth ')

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

model = Net()
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
writer.export_scalars_to_json("./log/stn.json")
writer.close()
'''
testm()


visualize_stn(5)
#plt.ioff()

