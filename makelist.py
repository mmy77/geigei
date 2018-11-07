

#!/usr/bin/env python  
import os

import random



def writeFile(List,tfilename):
	filtera = 'back'
	f = open(tfilename,'w')
	linename = []
	for line in List:
	    if line.find(filtera) >= 0:
	    	continue
	    linename.append(line)
	#print(linename,'##############')
	#linename.sort()

	for line in linename:
		#f.write(line+'\n')
		#print(line)
		f.write(line)
	print("order writing done!")
	f.close()

def shufflefile(filename):
	length = len('/mnt/ftp/datasets/dataset-gait/OU-ISIR/TreadmillDatasetB/00001M4000A0/')
	fh = open(filename, 'r')
	newlines = []
	block = []
	count = 0
	print("out order!")
	for line in fh:
		line = line.strip('\n')
		if count==0:
			nowname = line[:length]
		if line[:length]==nowname:
			block.append(line)
			count = count+1
		else:
			count = 0
			newlines.append(block)
			block = []
			block.append(line)
			count = 1
			nowname = line[:length]
	newlines.append(block)
	fh.close()	
	f = open(filename,'w')
	random.shuffle(newlines)
	for block in newlines:
		for line in block:
			#print(line)
			f.write(line+'\n')


def shufflefile2(filename):

	fh = open(filename,'r')
	lines = []
	for line in fh:
		lines.append(line)
	random.shuffle(lines)
	f = open(filename,'w')
	for line in lines:
		f.write(line)



def crop_file(filename):
	count = 0 
	remove = 0
	newlines = []
	dirname = ''
	length = len('/mnt/ftp/datasets/dataset-gait/TUMGAID/TUMGAIDimagetracked/croppedFrames/p001/b01/')
	temp16 = []
	tempj = 0
	
	fh = open(filename, 'r')
	for line in fh:
		line = line.strip('\n')

		if (tempj == 0 ):
			dirname = line[:length]
		if(line[:length]==dirname):
			tempj = tempj+1
			temp16.append(line)
			if(tempj==16):
				newlines = newlines + temp16
				temp16=[]
				tempj = 0
		else:
			tempj=0
			temp16=[]
	fh.close()
	f = open(filename,'w')
	for line in newlines:
		f.write(line+'\n')




def readfile(imgpath):
	imglist = os.listdir(imgpath)
	count = 0
	for filename in imglist:

		if filename[-3:]=='png' or filename[-3:]=='jpg':
				filename2 = filename.split('_')
				angle = filename2[-2] #090
				#if angle == '090':

				label = filename2[0]  #/mnt/ftp/datasets/dataset-gait/OU-ISIR/TreadmillDatasetB/00001M4000A0/
				#print(label)
				label_int = int(label)
				
				#print(label,label_int)
				#print(label_int)
				if label_int<=400:
					count = count + 1
					trainlist.append(filename+ ' ' + str(label_int))
					print(filename+ ' ' + str(label_int))
					#print(dirpath+'/'+filename+' '+str(label_int))
				#if label_int >250 and label_int<=250:
				#	vlalist.append(dirpath+'/'+filename+' '+str(label_int))
				if label_int >400:
					testlist.append(filename+ ' ' + str(label_int))
		if(count>=500):
			break
	return trainlist,testlist						

							#	count = count+1
							#	if count>100:
							#		break
	#	if count>100:
def read_file2(imgpath,respath):
	count = 0
	for (dirpath,dirnames,filenames) in os.walk(imgpath):
		for filename in filenames:
			if filename[-3:]=='png' or filename[-3:]=='jpg':
				label = dirpath[locate:]  #'/mnt/ftp/datasets/dataset-gait/CASIA/DatasetB/silhouettes/'#001/bg-01/000/'		
				#label = label.replace('/','_')
				#filename_label = str(int(filename[-7:-4])/20)
				#resfile = respath+label+'_'+filename_label+'.png'
				resfile = respath+label+'/'+filename
				#print(resfile)
				count = count + 1
				if os.path.exists(resfile):
					print(resfile)
					trainlist.append(dirpath+'/'+filename + ' '+resfile)
		if(count>10000):
			break
	return trainlist
def read_file3(imgpath):
	#count = 0
	for (dirpath,dirnames,filenames) in os.walk(imgpath):
		for filename in filenames:
			if filename[-3:]=='png' or filename[-3:]=='jpg':
				label = dirpath[locate:locate+3]  #'/mnt/ftp/datasets/dataset-gait/CASIA/DatasetB/silhouettes/'#001/bg-01/000/'		
				#count = count + 1
				trainlist.append(dirpath+'/'+filename+' '+str(int(label)))
				#continue
		#if(count>10000):
		#	break
	return trainlist

def setfile(txtfile): #split img into bolck
	f = open(txtfile, 'r')
	imgs = f.readlines()
	#imgs = imgs[491994:]
	predir = ''
	count = 0
	fileblock = []
	tmpblock = []
	for pairsss in imgs:
		pairs = pairsss.split()
		
		dirpath = pairs[0][:locate]
		if(dirpath!=predir):
			predir = dirpath
			if(len(tmpblock)!=0):
				fileblock.append(tmpblock)
			tmpblock = []
		tmpblock.append(pairsss)
	fileblock.append(tmpblock)
	random.shuffle(fileblock)
	trainlist = []
	for block in fileblock:
		count = 0
		for item in block:
			trainlist.append(str(count)+' '+item)
			count = count + 1
			#print(count)
	return trainlist

trainfile = 'datalist/train_all.txt'
newtrain = 'datalist/train_sgei.txt'
testfile = 'datalist/test_sgei.txt'
#imgpath = '/mnt/ftp/datasets/dataset-gait/CASIA/DatasetB/silhouettes/'#001/bg-01/000/'
imgpath = '/mnt/ftp/datasets/dataset-gait/CASIA/DatasetB/silhouettes/001/bg-01/090/'
respath = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/cropImg/'
locate = len(imgpath)
trainlist = []
vlalist = []
testlist = []
#trainlist = read_file3(imgpath)
trainlist = setfile('datalist/alltrain.txt')

writeFile(trainlist,trainfile)
#shufflefile2(trainfile)


txtfile = open(trainfile,'r')
imgname = txtfile.readlines()
length = len(imgname)
train_num = int(length*0.8)
trainame = imgname[:train_num]
testname = imgname[train_num:]
writefile = open(testfile,'w')
for name in testname:
	writefile.write(name)
f = open(newtrain,'w')
for name in trainame:
	f.write(name)
'''
'''

#print(trainlist)
#writeFile(vlalist,valfile)
#writeFile(testlist,testfile)
#shufflefile(trainfile)
#shufflefile(testfile)

#crop_file(trainfile)
#crop_file(testfile)
#crop_file(valfile)
#path 1
#path 2

'''

import matplotlib.pyplot as plt
logfile = './log/rrandom_optimpre199_5cropcross.txt'
f = open(logfile,'r')
epoch = []
loss = []
acc = []
#static = 9900
index = 1
for line in f:
	#if index>static:
	#	break
	#if index%50==0:
			
	a = line.strip('\n').split(' ')
	epoch.append(int(a[0]))
	loss.append(float(a[1]))
	acc.append(float(a[2]))
	#index = index + 1
	#print(epoch, '#',loss)

fig  =plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(epoch,loss,'red')
ax1.set_ylim([0,10])
ax1.set_ylabel('training loss')

ax2 = ax1.twinx()
ax2.plot(epoch,acc,'blue')
ax2.set_ylabel('training accuracy')
ax2.set_ylim([0,1])
ax2.set_xlabel('iterations')

plt.plot(epoch,loss)
plt.xlabel('ita')
plt.ylabel('acc')
plt.show()
'''