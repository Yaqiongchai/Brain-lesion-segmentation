import imgio as ig
import numpy as np
import getPatch2D as gp
import unet2D_updated as u
import tensorflow as tf
import gc
from sklearn import preprocessing

t2t1,_ = ig.load('ADA_histPadAXI.nii.gz')
t2t2,_ = ig.load('AOO_histPadAXI.nii.gz')
t2t3,_ = ig.load('BDB_histPadAXI.nii.gz')
t2t4,_ = ig.load('FEB_histPadAXI.nii.gz')
t2t5,_ = ig.load('GIN_histPadAXI.nii.gz')
t2t6,_ = ig.load('NBO_histPadAXI.nii.gz')
t2t7,_ = ig.load('RIM_histPadAXI.nii.gz')
t2t8,_ = ig.load('SNA_histPadAXI.nii.gz')
t2t9,_ = ig.load('WIJ_histPadAXI.nii.gz')
l1,_ = ig.load('ADA_GT_PadAXI.nii.gz')
l2,_ = ig.load('AOO_GT_PadAXI.nii.gz')
l3,_ = ig.load('BDB_GT_PadAXI.nii.gz')
l4,_ = ig.load('FEB_GT_PadAXI.nii.gz')
l5,_ = ig.load('GIN_GT_PadAXI.nii.gz')
l6,_ = ig.load('NBO_GT_PadAXI.nii.gz')
l7,_ = ig.load('RIM_GT_PadAXI.nii.gz')
l8,_ = ig.load('SNA_GT_PadAXI.nii.gz')
l9,_ = ig.load('WIJ_GT_PadAXI.nii.gz')

wfeature=[]
wlabel=[]

wfeature.append(gp.make_blocksAXI_norm(t2t1,8))
wfeature.append(gp.make_blocksAXI_norm(t2t2,8))
wfeature.append(gp.make_blocksAXI_norm(t2t3,8))
wfeature.append(gp.make_blocksAXI_norm(t2t4,8))
wfeature.append(gp.make_blocksAXI_norm(t2t5,8))
wfeature.append(gp.make_blocksAXI_norm(t2t6,8))
wfeature.append(gp.make_blocksAXI_norm(t2t7,8))
wfeature.append(gp.make_blocksAXI_norm(t2t8,8))
wfeature.append(gp.make_blocksAXI_norm(t2t9,8))
wlabel.append(gp.make_blocksAXI(l1,8))
wlabel.append(gp.make_blocksAXI(l2,8))
wlabel.append(gp.make_blocksAXI(l3,8))
wlabel.append(gp.make_blocksAXI(l4,8))
wlabel.append(gp.make_blocksAXI(l5,8))
wlabel.append(gp.make_blocksAXI(l6,8))
wlabel.append(gp.make_blocksAXI(l7,8))
wlabel.append(gp.make_blocksAXI(l8,8))
wlabel.append(gp.make_blocksAXI(l9,8))

feature=[]
label=[]
for a in range(0,9):
	for b in range(0,65536):
		if np.sum(wlabel[a][b])!=0:
			feature.append([wfeature[a][b]])
			label.append([wlabel[a][b]])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = u.unet_model_2d(input_shape=tuple([1,8,8]), n_labels=1, initial_learning_rate=0.1, deconvolution=True)
model.fit(feature,label,epochs=3000)

model.save('ofcn8_AXI_normNov25.h5')

