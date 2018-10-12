

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
	linename.sort()

	for line in linename:
		f.write(line+'\n')
		#print(line)
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


trainfile = 'train_all.txt'
newtrain = 'train_angle.txt'
testfile = 'test_angle.txt'
imgpath = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/GEI/'
locate = len(imgpath)
trainlist = []
vlalist = []
testlist = []


imglist = os.listdir(imgpath)
for filename in imglist:

	if filename[-3:]=='png' or filename[-3:]=='jpg':
			#count = count + 1
			#print(dirpath)
			#print ('filename',filename)
			#print ('dirname:',dirnames)
			filename2 = filename.split('_')
			angle = filename2[-2] #090
			#if angle == '090':

			label = filename2[0]  #/mnt/ftp/datasets/dataset-gait/OU-ISIR/TreadmillDatasetB/00001M4000A0/
			#print(label)
			label_int = int(label)
			
			#print(label,label_int)
			#print(label_int)
			if label_int<=400:
				trainlist.append(filename+ ' ' + str(label_int))
				print(filename+ ' ' + str(label_int))
				#print(dirpath+'/'+filename+' '+str(label_int))
			#if label_int >250 and label_int<=250:
			#	vlalist.append(dirpath+'/'+filename+' '+str(label_int))
			if label_int >400:
				testlist.append(filename+ ' ' + str(label_int))
						

						#	count = count+1
						#	if count>100:
						#		break
#	if count>100:
writeFile(trainlist,trainfile)
shufflefile2(trainfile)


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