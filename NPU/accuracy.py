import argparse
import os
import keras
from keras.utils.vis_utils import plot_model





#directory at which converting scripts are
script_dir="/home/ehsan/UvA/Sub_Model/"
quantized=True
#directory that images are there
quant_dataset_dir="/home/ehsan/UvA/Accuracy/Keras/Yolov3/keras-yolo3/validation_set/images/"
# dir for putting datasetlis text file
dataset_dir="/home/ehsan/UvA/Accuracy/Keras/Yolov3/datasets/"



import glob
'''if n == 0:
		print(f'There are {len(files)} files in the directory.')
		n=len(files)'''
def quant_dataset_list(_dir,n=0):
	files=glob.glob(f'{_dir}/*')
	if len(files) < n: 
		print(f'There are only {len(files)} files in the directory.')
		n=len(files)
	if n == 0:
		print(f'There are {len(files)} files in the directory.')
		n=len(files)
	file_name=f'{dataset_dir}/dataset_{n}'
	f=open(file_name,'w')
	for img in files:
		print(img)
		f.write(img+'\n')
	return file_name



if __name__ == "__main__":
	# parsing input arguments
	parser = argparse.ArgumentParser(description='Slice a model')
	parser.add_argument('--Model', metavar='path', required=False,
						help='Model')
	parser.add_argument('--Structure', metavar='path', required=False,
						help='Structure of the model (prototxt)')
	
	args = parser.parse_args()
	##########################
	model_dir=os.path.dirname(args.Model)
	#Keras Models:
	if args.Model.split('.')[-1]=='h5':
		#load the model
		model=keras.models.load_model(args.Model)
		plot_model(model, to_file='./model_plot.png', show_shapes=False, show_layer_names=True)

		#set the model parameters
		pb_convert_args={}
		pb_convert_args["input"]=model.layers[0].name
		try:
			pb_convert_args["output"]=model.layers[-1].get_output_at(0).op.name
		except:
			print('Output tensor of last layer has no op attribute try to directly get name')
			pb_convert_args["output"]=model.layers[-1].get_output_at(0).name       
		input_shape=''
		s=list(model.layers[0].get_input_shape_at(0)[1:])
		for x in s:
			input_shape=input_shape+str(x)+','
		input_shape=input_shape[:-1]
		pb_convert_args["input_shape"]=input_shape
		pb_convert_args["input_shape"]='608,608,3'
		pb_convert_args["h5name"]=args.Model
		pb_convert_args["pb_name"]=args.Model.replace('.h5','.pb')
		print(f'Model parameters for converting are:{pb_convert_args}')
		####

	
		#Convert from keras to tensorflow(pb)
		cmd1=f'conda run -n rock-kit3 python {script_dir}keras_to_tensorflow.py --input_model={pb_convert_args["h5name"]} --output_model={pb_convert_args["pb_name"]}'
		print(f'command is: {cmd1}')
		rcode=os.system(cmd1)
		print(f'Freezing graph return code is {rcode} ')
		if rcode!=0:
			exit
		##

		#convert from pb to rknn
		#if quantized:
		dataset_list=quant_dataset_list(quant_dataset_dir,0)
		if quantized:
			cmd2=f'conda run -n rock-kit3 python {script_dir}convert.py {pb_convert_args["pb_name"]} {pb_convert_args["input"]} {pb_convert_args["output"]} {pb_convert_args["input_shape"]} {1} {dataset_list}'
		else:
			cmd2=f'conda run -n rock-kit3 python {script_dir}convert.py {pb_convert_args["pb_name"]} {pb_convert_args["input"]} {pb_convert_args["output"]} {pb_convert_args["input_shape"]}'
		print(f'command is: {cmd2}')
		ok=os.system(cmd2)
		print(f'Convert to rknn return code is {ok} ')
		##

