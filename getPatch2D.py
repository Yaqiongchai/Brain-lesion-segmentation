import numpy as np
from sklearn import preprocessing
# for AXI image, img1.shape[2] is the iteration direction (+1 instead of +t)
def make_blocksAXI(image,t):
	x = range(0,image.shape[1],t)
	y = range(0,image.shape[2],1)
	z = range(0,image.shape[0],t)
	reshaped = []
	for c in z:
		for d in x:
			for a in y:
				tmp1=image[c:c+t,d:d+t,a]
				reshaped.append(tmp1.reshape(t,t))
				#reshaped.append(image[c:c+t,d:d+t,a:a+1])
	return reshaped
# for AXI image, img1.shape[0] is the iteration direction (+1 instead of +t)
def make_blocksAXI_norm(image,t):
        x = range(0,image.shape[1],t)
        y = range(0,image.shape[2],1)
        z = range(0,image.shape[0],t)
        reshaped = []
        for c in z:
                for d in x:
                        for a in y:
                                tmp1=image[c:c+t,d:d+t,a]
                                imageTP=np.reshape(tmp1,(t*t,1))
                                imageNTP=preprocessing.scale(imageTP)
                                imageNP=np.reshape(imageNTP,(t,t))
                                reshaped.append(imageNP)
        return reshaped
def make_blocksCOR_norm(image,t):
        x = range(0,image.shape[1],1)
        y = range(0,image.shape[2],t)
        z = range(0,image.shape[0],t)
        reshaped = []
        for c in z:
                for d in x:
                        for a in y:
                                tmp1=image[c:c+t,d,a:a+t]
                                tmp1.shape
                                imageTP=np.reshape(tmp1,(t*t,1)) 
                                imageNTP=preprocessing.scale(imageTP)
                                imageNP=np.reshape(imageNTP,(t,t))
                                reshaped.append(imageNP)
        return reshaped                        
def make_blocksCOR(image,t):
	x = range(0,image.shape[1],1)
	y = range(0,image.shape[2],t)
	z = range(0,image.shape[0],t)
	reshaped = []
	for c in z:
		for d in x:
			for a in y:
                                tmp1=image[c:c+t,d,a:a+t]
                                reshaped.append(tmp1.reshape(t,t))
	return reshaped
def combine_blocks(imageList, x, y, z, t):
        combined = np.zeros(shape=(x,y,z))
        xr = range(0, x, t)
        yr = range(0, y, t)
        zr = range(0, z, 1)
        for x1 in xr:
                 for y1 in yr:
                         for z1 in zr:
                                tmp1=imageList[int((x1*y*z)/(t*t)+(y1*z)/t+z1)]
                                combined[x1:x1+t,y1:y1+t,z1]=np.reshape(tmp1,(t,t))
        return combined
def combine_blocks_cor(imageList,x,y,z,t):
        combined = np.zeros(shape=(x,y,z))
        xr = range(0, x, t)
        yr = range(0, y, 1)
        zr = range(0, z, t)
        for x1 in xr:
                 for z1 in zr:
                         for y1 in yr:
                                tmp1=imageList[int((x1*y*z)/(t*t)+(z1*y)/t+y1)]
                                combined[x1:x1+t,y1,z1:z1+t] = np.reshape(tmp1,(t,t))
        return combined
def make_blocks_nor(image,t):
	x = range(0,image.shape[1],t)
	y = range(0,image.shape[2],t)
	z = range(0,image.shape[0],t)
	reshaped = []
	for c in z:
		for d in x:
			for a in y:
				imageP = image[c:c+t,d:d+t,a:a+t]
				imageTP = np.reshape(imageP,(t*t*t,1))
				imageNTP = preprocessing.scale(imageTP)
				imageNP = np.reshape(imageNTP,(t,t,t))
				reshaped.append(imageNP)
	return reshaped

def make_dblocks_nor(image,t):
	x = range(0,image.shape[1])
	y = range(0,image.shape[2])
	z = range(0,image.shape[0])
	reshaped = []
	for c in range(len(z)-t+1):
		for d in range(len(x)-t+1):
			for a in range(len(y)-t+1):
				imageP = image[c:c+t,d:d+t,a:a+t]
				imageTP = np.reshape(imageP,(t*t*t,1))
				imageNTP = preprocessing.scale(imageTP)
				imageNP = np.reshape(imageNTP,(t,t,t))
				reshaped.append(imageNP)
	return reshaped

def make_dblocks(image,t):
	x = range(0,image.shape[1])
	y = range(0,image.shape[2])
	z = range(0,image.shape[0])
	reshaped = []
	for c in range(len(z)-t+1):
		for d in range(len(x)-t+1):
			for a in range(len(y)-t+1):
				reshaped.append(image[c:c+t,d:d+t,a:a+t])
	return reshaped

def combine_dblocks(imageList, x, y, z, t):
	combined = np.zeros(shape=(x,y,z))
	xr = range(0, x)
	yr = range(0, y)
	zr = range(0, z)
	for z1 in range(z-t+1):
		for y1 in range(y-t+1):
			for x1 in range(x-t+1):
				addm = np.zeros(shape=(x,y,z))
				addm[z1:z1+t,y1:y1+t,x1:x1+t]=imageList[int((z1*(x-t+1)*(y-t+1))+(y1*(x-t+1))+x1)]
				combined = combined + addm
	return combined
