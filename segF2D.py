from keras.models import load_model
from keras import backend as K
from unet import dice_coef,dice_coef_loss
from deconvolutional import Deconvolution2D
import imgio as ig
import numpy as np
import getPatch2D as gp
import nibabel as nib
import tensorflow as tf
from sklearn import preprocessing

def seg8(filename,patch_size):
	Img,affine = ig.load(filename)
	imgList = gp.make_blocksAXI_norm(Img,patch_size)
	newImg = []
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	model = load_model('ofcn8_AXI_normNov25.h5',custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'Deconvolution2D':Deconvolution2D})
	for i in range(0,65536):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		l = model.predict(np.array([[imgList[i]]], np.float32))
		newImg.append(l[0][0])
		print('patch '+str(i)+' is done!')
	newI = gp.combine_blocks(newImg,256,256,64,patch_size)
	new_i = nib.Nifti1Image(newI,affine)
	nib.save(new_i, filename[:14]+'_Seg8_AXI.nii.gz')
	return 1

def seg16(filename,patch_size):
	Img,affine = ig.load(filename)
	imgList = gp.make_blocksAXI_norm(Img, patch_size)
	newImg = []
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	model = load_model('ofcn16_AXINov25_norm.h5',custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'Deconvolution2D':Deconvolution2D})
	for i in range(0,16384):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		l = model.predict(np.array([[imgList[i]]], np.float32))
		newImg.append(l[0][0])
		print('patch '+str(i)+' is done!')
	newI = gp.combine_blocks(newImg,256,256,64,patch_size)
	new_i = nib.Nifti1Image(newI,affine)
	nib.save(new_i, filename[:14]+'_Seg16_AXI.nii.gz')
	return 1


def seg32(filename,patch_size):
	Img,affine = ig.load(filename)
	imgList = gp.make_blocksAXI_norm(Img, 32)
	newImg = []
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	model = load_model('MD32_AXINOV25_norm.h5',custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'Deconvolution2D':Deconvolution2D})
	for i in range(0,4096):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		l = model.predict(np.array([[imgList[i]]], np.float32))
		newImg.append(l[0][0])
		print('patch '+str(i)+' is done!')
	newI = gp.combine_blocks(newImg,256,256,64,patch_size)
	new_i = nib.Nifti1Image(newI,affine)
	nib.save(new_i, filename[:14]+'_Seg32_AXI.nii.gz')
	return 1
def seg64(filename,patch_size):
	Img,affine = ig.load(filename)
	imgList = gp.make_blocksAXI_norm(Img, patch_size)
	newImg = []
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	model = load_model('MD64_AXI_normNOV25.h5',custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'Deconvolution2D':Deconvolution2D})
	for i in range(0,1024):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		l = model.predict(np.array([[imgList[i]]], np.float32))
		newImg.append(l[0][0])
		print('patch '+str(i)+' is done!')
	newI = gp.combine_blocks(newImg,256,256,64,patch_size)
	new_i = nib.Nifti1Image(newI,affine)
	nib.save(new_i, filename[:14]+'_Seg64_AXI.nii.gz')
	return 1

def seg128(filename,patch_size):
	Img,affine = ig.load(filename)
	imgList = gp.make_blocksAXI_norm(Img, patch_size)
	newImg = []
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	model = load_model('model128_2DsitNOV24.h5',custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'Deconvolution2D':Deconvolution2D})
	for i in range(0,256):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		l = model.predict(np.array([[imgList[i]]], np.float32))
		newImg.append(l[0][0])
		print('patch '+str(i)+' is done!')
	newI = gp.combine_blocks(newImg,256,256,64,patch_size)
	new_i = nib.Nifti1Image(newI,affine)
	nib.save(new_i, filename[:14]+'_Seg128_AXI.nii.gz')
	return 1
