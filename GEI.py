import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#imgpath_origin = '/home/xiaoqian/pseudo-3d-pytorch/example/img/'

#imgnames = os.listdir(imgpath)
def mktxt(imgpath):


	txtfile = 'silhouette.txt'
	f = open(txtfile,'w')
	filelabel = 0
	
	for (dirpath,dirnames,filenames) in os.walk(imgpath):
		filenames.sort()
		filelabel = filelabel + 1
		print(dirpath)			
		for imgname in filenames:
			if(imgname[-3:]=='jpg' or imgname[-3:]=='png'):
				f.write(dirpath+' '+imgname + '\n')
	f.close()



def mkGEI(imgsblock,order,imgpath,f_Gei):#start-end
	crop_width = 88
	crop_height = 128
	period = 0
	imgdata_all = np.zeros((128,88))
	for pairs in imgsblock:
		#pairs = pairs.split()
		dirpath = pairs[0]
		imgname = pairs[1]
		#print(pairs)
		#continue
		#print(imgname)
		x_index = []
		y_index = []
		imgfile = Image.open(dirpath+'/'+imgname)
		#print(dirpath+'/'+imgname)
		imgdata = np.array(imgfile)
		#imgdata = imgdata[:,:,0]
		#print(imgdata.shape)
		height,width = imgdata.shape #320x240

		for i in range(height):
			for j in range(width):
				if imgdata[i,j]>0:
					x_index.append(i)
					y_index.append(j)  #y--width--rows
		if(len(x_index)<=0 or len(y_index)<=0):
			continue
		x_min = float(min(x_index))
		x_max = max(x_index)
		y_min = float(min(y_index))
		y_max = max(y_index)
		#print(x_max,x_min,y_max,y_min)
		if (x_max-x_min)/height<=0.05:
			continue
		#ratio.append((x_max-x_min)/(y_max-y_min)  )
		crop_height = x_max - x_min
		resize_ratio = crop_height / 128.0
		crop_width = 88 * resize_ratio
		
		y_min = y_min - (crop_width/2 -(y_max-y_min)/2.0)
		y_max = y_min+crop_width
		x_min = x_min - (crop_height/2-(x_max-x_min)/2.0)
		x_max = x_min + crop_height

		cropped = imgfile.crop((y_min,x_min,y_max,x_max))
		cropped = cropped.resize((88,128)) # width, height
		#cropped.save(dirpath+'/crop_'+imgname)
		imgdata = np.array(cropped)
		#imgdata = imgdata[:,:,0]
		imgdata_all = imgdata_all+imgdata

		period = period + 1
	imgdata_all=imgdata_all*255/(imgdata_all.max())
	img = Image.fromarray(imgdata_all)
	img = img.convert('L')
	#img.show()
	#print(dirpath)
	length = len(imgpath)
	savepath = resultpath + dirpath[length:].replace('/','_')+'_'+str(order)+'.png'
	print(savepath)
	img.save(savepath)
	

	f_Gei.write(savepath + ' ' + str(int(dirpath[length+1:length+4])))

	#return imgdata_all

def clipimgs(txtfile): #split img into bolck
	f = open(txtfile, 'r')
	imgs = f.readlines()
	#imgs = imgs[491994:]
	predir = ''
	fileblock = []
	tmpblock = []
	for pairs in imgs:
		pairs = pairs.split()
		
		dirpath = pairs[0]
		if(dirpath!=predir):
			predir = dirpath
			if(len(tmpblock)!=0):
				fileblock.append(tmpblock)
			tmpblock = []
		tmpblock.append(pairs)
	fileblock.append(tmpblock)
	return fileblock


imgpath = '/home/lxq/PSPNet/evaluation/psp/color/'
#imgpath = '/home/lxq/pseudo-3d-pytorch/example/090/'
#resultpath = '/home/lxq/pseudo-3d-pytorch/example/casia-align/'
resultpath = '/home/lxq/GeiGait/data/'
length = len(imgpath)
txtfile = 'silhouette.txt'
#mktxt(imgpath)
print('txt done!')
fileblock = clipimgs(txtfile)
print('clip done!')
GeiTxt = 'geilist.txt'
f_Gei = open(GeiTxt,'w')

for imgs in fileblock: # imgs splited
	length = len(imgs) # the num of imgs in a file
	croplength = 20
	cropnum = 10 # each sequence generate 10 gei
	interval = (length-croplength)/(cropnum-1)
	if(interval<5): #aviod the interval too small
		interval = 5
	print("interval:",interval,"length:",length)
	GEIsequence = []
	for i in range(0,len(imgs)-croplength+1,interval):
		GEIsequence = []
		if(i>=length or  i+croplength>=length):
			break
		for j in range(i,i+croplength):
			#print(imgs[j])
			GEIsequence.append(imgs[j])
		mkGEI(GEIsequence,i/interval,imgpath,f_Gei)

f_Gei.close()


