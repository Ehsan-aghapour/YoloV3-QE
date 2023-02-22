#Alexnet:
#time python rknn_inference.py /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/AlexNet/bvlc_alexnet/new/deploy.prototxt /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/AlexNet/bvlc_alexnet/new/bvlc_alexnet.caffemodel

#Googlenet:
#time python rknn_inference.py /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/GoogleNet/bvlc_googlenet/new/deploy.prototxt /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/GoogleNet/bvlc_googlenet/new/bvlc_googlenet.caffemodel

#Squeezenet:
#time python rknn_inference.py /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/SqueezeNet-master/SqueezeNet_v1.0/new/deploy.prototxt /home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/SqueezeNet-master/SqueezeNet_v1.0/new/squeezenet_v1.0.caffemodel

import numpy as np
import cv2
from rknn.api import RKNN
import sys
from PIL import Image

from scipy.ndimage import zoom
from skimage.transform import resize
#conda install scikit-image
#import caffe


import keras_load_img as ld


# explicit_mean_reduction means that you want to do mean reduction explicitly
# explicit_channel_reorder means that you want to do channel reordering of caffe models explicitly
# otherwise in npu rknn.config these operations should be done 
# for F16 set explicit_mean_reduction and explicit_channel_reorder to false and set quant to false (True is for U8)
# (it is also ok to set both to true which explicitly do mean reducion and channel reordering)
# for U8 set both to False and do not forget to set quant to True
# (it is possible to set them to false both the quantization dataset should be preprocessed in terms of 
# mean reduction and channel reordering and so save them as proper size in npy format --> NPY=true)
explicit_mean_reduction=False
explicit_channel_reorder=False

# this is for quantization dataset if are .npy (resized and mean reduced and channels are compatible with model)
# otherwise it is png images that resized (if npy=flase explicit variables should be set to false)
NPY=False

quant=True
precompile=False
PC=False

#Alexnet and Squeezenet:
#Input_size=227

#Others:
Input_size=608






################################################ From keras preprocessing (accuracy.py) ######################

nh=Input_size
nw=Input_size
batch_size=100
n=50000

networks=['mobilenet','resnet50','Yolov3']
network=networks[2]

models_dir='/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/models/'
acc_dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Accuracy/"
img_dir=acc_dir+'/Imagenet/'



#MobileNet:
if network==networks[0]:
	m='MobileNet/MobileNet.pb'
	inputs='input_2'
	outputs='act_softmax/Softmax'
	INPUT_SIZE=[224,224,3]

#ResNet50:
if network==networks[1]:
	m='Resnet50/ResNet50.pb'
	inputs='input_1'
	outputs='fc1000/Softmax'
	INPUT_SIZE=[224,224,3]




model_dir=models_dir+m
if len(sys.argv) > 1:
	name=sys.argv[1]
else:
	name=model_dir
print(f'Model name:{name}')
rknn_name=name.split('/')[-1].split('.')[0]+'.rknn'
csv_name=name.split('/')[-1].split('.')[0]+'.csv'
rknn_name_precompiled=name.split('/')[-1].split('.')[0]+'_precompiled.rknn'
model_type=name.split('.')[-1]

# input: image number, outout: image name (iamgenet)
def image_name(im_n):
    im_n=str(im_n).zfill(8)
    im_n=im_n
    img_name='ILSVRC2012_val_'+im_n+'.JPEG'
    return img_dir+'/ILSVRC2012_img_val/'+img_name


mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
#mean = np.array([ 104.01, 116.67, 122.68 ], dtype=np.float32)
'''
def prepare_image(file):
    img = image.load_img(file, target_size=(nh, nw))
    img_array = image.img_to_array(img)
    #img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    if m=='Resnet50/ResNet50.pb':
        return keras.applications.resnet50.preprocess_input(img_array)
        #return keras.applications.resnet50.preprocess_input(img_array,data_format=None,mode='caffe')
        #return imagenet_utils.preprocess_input(img_array, data_format=None, mode='caffe')
    if m=='MobileNet/MobileNet.pb':
        return keras.applications.mobilenet.preprocess_input(img_array)
'''

def prepare_image_2(file):
	img = ld.load_img(file, target_size=(nh, nw))
	img=np.array(img,dtype="float32")
	#print(img.shape)
	#img=np.expand_dims(img,axis=0)
	#ToDo
	#print(m)
	if m=='Resnet50/ResNet50.pb':
		if explicit_channel_reorder==True:
			img = img[..., ::-1]
		if explicit_mean_reduction==True:
			img=img-mean
		#return img
        #img[..., 0] -= mean[0]
        #img[..., 1] -= mean[1]
        #img[..., 2] -= mean[2]
        #return img
	if m=='MobileNet/MobileNet.pb':
		if explicit_mean_reduction:
			img /= 127.5
			img -= 1.
		#return img

	return img

############################################################################################




	
def dataset(i,j):
	prefix=img_dir+'/ILSVRC2012_img_val_keras_resized_PNG_'+str(Input_size)+'/'
	if NPY:
		prefix=img_dir+'/keras_preprocessed_'+str(Input_size)+'/'
	#_resized
	file_name=f'./dataset_{i}_{j}'
	f=open(file_name,'w')
	for im_n in range(i,j+1):
		im_n=str(im_n).zfill(8)
		im_n='ILSVRC2012_val_'+im_n
		if NPY:
			im_n=im_n+'.npy'
		else:		
			im_n=im_n+'.PNG'
		print(im_n)
		im=prefix+im_n
		f.write(im+'\n')
		
	return file_name


import glob

def quant_dataset_list(n,_dir):
	files=glob.glob(f'{_dir}/*')
	if len(files) < n :
		print(f'There is only {len(files)} files in the directory.')
		n=len(files)
	file_name=f'./dataset_{n}'
	f=open(file_name,'w')
	for img in files:
		print(_dir+img)
		f.write(_dir+img+'\n')
	return file_name


def show_outputs(outputs):
	output = outputs[0][0]
	output_sorted = sorted(output, reverse=True)
	top5_str = 'mobilenet_v1\n-----TOP 5-----\n'
	for i in range(5):
		value = output_sorted[i]
		index = np.where(output == value)
		for j in range(len(index)):
			if (i + j) >= 5:
				break
			if value > 0:
				topi = '{}: {}\n'.format(index[j], value)
			else:
				topi = '-1: 0.0\n'
			top5_str += topi
	print(top5_str)

def show_perfs(perfs):
	perfs = 'perfs: {}\n'.format(outputs)
	print(perfs)


if __name__ == '__main__':

	# Create RKNN object
	rknn = RKNN()
	
	
	# init runtime environment
	print('--> Init runtime environment')
	
	if model_type!='rknn':
		# pre-process config
		print('--> config model')
		#rknn.config(channel_mean_value='103.94 116.78 123.68 58.82', reorder_channel='0 1 2')
		#rknn.config(channel_mean_value='103.94 116.78 123.68 1', reorder_channel='2 1 0')
		if explicit_channel_reorder or network=='mobilenet':
			print("Using RGB channel order in network")
			r_ch='0 1 2'
		else:
			print("Using BGR channel order in network")
			r_ch='2 1 0'
		
		if explicit_mean_reduction:
			ch_m='0 0 0 1'
		else:
			if network=='mobilenet':
				ch_m='127.5 127.5 127.5 127.5'
				
			else:
				if network=='resnet50':
					ch_m='103.939 116.779 123.68 1'
				else:
					ch_m='104.01 116.67 122.68 1'
		
		print(f'Using channel mean {ch_m} in network')
		rknn.config(channel_mean_value=ch_m, reorder_channel=r_ch)
			
		

		print('done')

		# Load tensorflow model
		print('--> Loading model')
		#ret = rknn.load_tflite(model='./mobilenet_v1.tflite')
		print('--> Loading model')
	#rknn.load_tensorflow(tf_pb='model.pb',
	#                     inputs=['test_in'],
	#                     outputs=['test_out/BiasAdd'],
	#                     input_size_list=[[INPUT_SIZE]])
	#rknn.load_onnx(name)
		if model_type=='pb':
			if len(sys.argv)==5:
				inputs=sys.argv[2]
				outputs=sys.argv[3]
				INPUT_SIZE=sys.argv[4]
			rknn.load_tensorflow(tf_pb=name,
				inputs=[inputs],
				outputs=[outputs],
				input_size_list=[INPUT_SIZE])
				

		if model_type=='onnx':
			rknn.load_onnx(name)

		if model_type=='prototxt':
			#p=Path(name)
			#proto_name=p.with_suffix('.prototxt')
			print(f'name:{name},blobs:{sys.argv[2]}')
			ret = rknn.load_caffe(model=name,
				proto='caffe',
				blobs=sys.argv[2])

		print('done')

		# Build model
		print('--> Building model')
		if quant:
			ret = rknn.build(do_quantization=True, dataset=dataset(1000,2000))
		else:
			ret = rknn.build(do_quantization=False)
		#ret = rknn.build(do_quantization=False)
		#ret = rknn.build(do_quantization=False,rknn_batch_size=10)
		if ret != 0:
			print('Build mobilenet_v1 failed!')
			exit(ret)
		print('done')

		# Export rknn model
		print('--> Export RKNN model')
		ret = rknn.export_rknn(rknn_name)
		if ret != 0:
			print(f'Export {rknn_name} failed!')
			exit(ret)
		print('done')
		exit(0)
		####ret = rknn.init_runtime(target='RK3399Pro',rknn2precompile=True)
		ret = rknn.init_runtime()
		if ret != 0:
			print('Init runtime environment failed')
			exit(ret)
		'''ret = rknn.export_rknn_precompile_model(rknn_name_precompiled)
		if ret != 0:
			print('export prcompile failed')
			exit(ret)'''
	else:
		#rknn.load_rknn('./mobilenet_v1_sample_test_precompiled.rknn')
		rknn.load_rknn(sys.argv[1])
		if 'precompiled' in sys.argv[1]:
			PC=0
		if PC:
			ret = rknn.init_runtime()
		else:
			ret = rknn.init_runtime(target='RK3399Pro')
		if ret != 0:
			print('Init runtime environment failed')
			exit(ret)
		print('done')

	# Inference
	print('--> Running model')

	
	n=50000
	l='labels.txt'
	label_names = np.loadtxt(l, str, delimiter='\t')
	f=open(csv_name,'w')
	for i in range(1,n+1):
		#out = rknn.inference(inputs=[read_resized_image_1(i)])
		
		print(f'{i}/{n}')
		#print(len(inputs[0]))
		out = rknn.inference(inputs=[prepare_image_2(image_name(i))])
		
		#print(type(out))-->list
		#print(out.shape)
		prob=np.array(out)
		#print(prob.shape)
		#prob = out['prob']
		prob = np.squeeze(prob)
		idx = np.argsort(-prob)

		lbl='labels.txt'
		label_names = np.loadtxt(lbl, str, delimiter='\t')
		for i in range(5):
			label = idx[i]
			#print('%d   %.2f - %s' % (idx[i],prob[label], label_names[label]))
			#print(label_names[label].split(' ')[0])
			
			f.write(label_names[label].split(' ')[0])
			if i==4:
				f.write('\n')
			else:
				f.write(',')
	
	
	
	#show_outputs(outputs)
	print('done')
	'''print(f'output shape:{np.array(outputs).shape}')
	p=0
	if p:
		for i,output in enumerate(outputs[0][0]):
			if (i+1)%10:
				print(f'{i:<4}:{output:^10.4f}',end='\t')
			else:
				print(f'{i:<4}:{output:^10.4f}')

	perf=0
	if perf:
		# perf
		print('--> Begin evaluate model performance')
		perf_results = rknn.eval_perf(inputs=[img])
		print('done')
	'''
	rknn.release()
	
	
	
	
	'''
	batch_size=10
	f=open('alex_rknn.csv','w')
	n=400
	last_i=1
	#l='imagenet_labels.txt'
	l='labels.txt'
	label_names = np.loadtxt(l, str, delimiter='\t')
	for indx in range(batch_size+1,n+batch_size+1,batch_size):
		print(f"start of batch with index {last_i} to {indx}")
		images=[]
		for j in range(last_i,indx):
			local_indx=((j-1)%batch_size)
			images.append(read_image_1(j))
		#input("first batch image read\n")
		out = rknn.inference(inputs=[np.array(images)])
		#print(type(out))
		#print(out.shape)
		prob=np.array(out)
		print(prob.shape)
		#prob = out['prob']
		prob = np.squeeze(prob)
		idx = np.argsort(-prob)
		#input("inference for first batch finished\n")
		for j in range(last_i,indx):
			local_indx=((j-1)%batch_size)
			for i in range(5):
				label = idx[local_indx][i]
				print('%d   %.2f - %s' % (idx[local_indx][i],prob[local_indx][label], label_names[label]))
				#print(label_names[label].split(' ')[0])
			
				f.write(label_names[label].split(' ')[0])
				if i==4:
					f.write('\n')
				else:
					f.write(',')
		#input("first batch was written\n")
		last_i=indx
	'''

