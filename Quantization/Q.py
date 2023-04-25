# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: qe
#     language: python
#     name: qe
# ---

# + [markdown] id="KimMZUVqcJ8_"
# ##### Copyright 2021 The TensorFlow Authors.

# + [markdown] id="BlWzg1D9_EhW"
# # Inspecting Quantization Errors with Quantization Debugger

# + [markdown] id="XLoHL19yb-a0"
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/lite/performance/quantization_debugger"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_debugger.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/quantization_debugger.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/tensorflow/tensorflow/lite/g3doc/performance/quantization_debugger.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
#   <td>
#     <a href="https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub model</a>
#   </td>
# </table>

# + [markdown] id="qTEEzJWo_iZ_"
# ### Setup
#
# This section prepares libraries, MobileNet v3 model, and test dataset of 100
# images.

# + id="l7epUDUP_6qo"
# #!jupyter nbconvert --to script Q.ipynb
# ###!jupytext --set-formats ipynb,py Q.py --sync
# Quantization debugger is available from TensorFlow 2.7.0
# !pip uninstall -y tensorflow
# !pip install tf-nightly
# #!pip install tensorflow_datasets --upgrade  # imagenet_v2 needs latest checksum
# #!pip install tensorflow_hub

# !pip install pandas
# !pip install matplotlib

# + id="LLsgiUZe_hIa"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow_datasets as tfds
#import tensorflow_hub as hub

from tensorflow.lite.python import convert
import pickle
import os
from tensorflow.lite.python import interpreter as interpreter_wrapper
import numpy as np
import time
import threading
import pandas as pd 
import sys
import os
import itertools
cur_dir=os.getcwd()
sys.path.append(cur_dir+'/../Evaluation/')
import eval_multiThread2 as tt
# -


'''df=pd.read_csv("df.csv",index_col=0)
df = pd.DataFrame(columns=["name","mAP"])
df.loc[0]=["a", 3.2]
df.loc[1]=["b", 4.1]
df.to_csv("test.csv")
df2=pd.read_csv("test.csv",index_col=0)
df2.loc[2]=["c", 5.1]
df2
df.loc[71]=["2",3]'''

# +
server=1
GPU=1

p="/home/ehsan/UvA/Accuracy/Keras/"
p_server="/home/ehsan/Accuracy/"
if server:
    p=p_server
data_dir = p+"YOLOV3/Dataset/val2017"
image_size = (608, 608)
N=300


resdir='Yolo_files/'
ModelName=resdir+'Yolov3.h5'
QuantizedName=resdir+'YoloV3_quztized.tflite'
QSelectiveName=resdir+'YoloV3_selective_quztized.tflite'
UQSelectiveName=resdir+'YoloV3_selective_unquztized.tflite'
RESULTS_FILE = resdir+'yolov3_debugger_results.csv'
RESULTS_FILE_ANALYZED = resdir+'yolov3_debugger_results_analyzed.csv'
RESULTS_FILE_Propogate = resdir+'yolov3_debugger_propogate_results.csv'
RESULTS_FILE_Propogate_ANALYZED = resdir+'yolov3_debugger_propogate_results_analyzed.csv'
DebuggerName=resdir+'Debugger_Yolov3.pkl'
DebuggerPropogateName=resdir+'Debugger_Yolov3_propogation.pkl'
CalibratedName=resdir+'YoloV3_calibrated.tflite'


# Define the input shape and data type
input_shape = (1, 608, 608, 3)
input_dtype = tf.float32

# Define the output shape and data type
output_shape = (1,)
output_dtype = tf.float32

if GPU:
    #tf.debugging.set_log_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)



# +
def load_model(m=ModelName):
    model = tf.keras.models.load_model(m)
    return model





# Define a function to load and preprocess each image
def preprocess_image(file_path):
    # Load the image
    image = tf.io.read_file(file_path)
    # Decode the JPEG image to a tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize the image to the desired size
    image = tf.image.resize(image, image_size)
    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0
    return image

#train_dataset = tf.data.Dataset.from_tensor_slices((images))
#train_dataset=train_dataset.map(process_image)
def load_dataset():
    # Create a list of file paths to the JPEG images
    file_paths = tf.data.Dataset.list_files(data_dir + "/*.jpg")
    # Use the map() method to apply the preprocessing function to each image
    dataset = file_paths.map(preprocess_image)
    #dataset = dataset.map(lambda x: {'input_1': x})
    return dataset



'''
def gen_rep():
    train_dataset=prepare_dataset()    
    representative_dataset = train_dataset.take(100).batch(1)
    return representative_dataset
    
def representative_dataset(dataset):
	def _data_gen():
		for data in dataset.batch(1):
			yield [data['image']]
	return _data_gen
'''



#representative_dataset = dataset.take(300).batch(1)
def rep(_dataset,n=N):
    def representative_dataset():
        for img in _dataset.take(n):
            #img = tf.cast(img, tf.float32)
            yield {'input_1': np.array([img])}
            #yield np.array(img)
    #return tf.data.Dataset.from_generator(representative_dataset, {'input_1': tf.float32}, {'input_1': tf.TensorShape([1, None, None, 3])})
    return representative_dataset

def rep2(_dataset,n=N):
    def representative_dataset():
        for img in _dataset.take(n):
            #img = tf.cast(img, tf.float32)
            
            #img = tf.expand_dims(img, axis=0)
            #yield [np.array(img)]
            
            yield [np.array([img])]
    return representative_dataset

def rep3(_dataset,n=N):
    for img in _dataset.take(n):
        yield [np.array([img])]

def quantize(model,_dataset,name=QuantizedName):
    print("\n\n\n\n***************************************************")
    print("quantization...\n")
    if False and (os.path.isfile(QuantizedName)):
        print(f"loading existed {QuantizedName}")
        with open(name, 'rb') as f:
            quantized_model=f.read()
    else:
        print(f'quantization: producing file {QuantizedName}')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.representative_dataset = tf.lite.RepresentativeDataset(rep(_dataset))
        converter.representative_dataset = rep2(_dataset,N)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.target_spec.supported_ops = [
        #    tf.lite.OpsSet.TFLITE_BUILTINS_INT8_GPU 
        #]
        #converter.inference_input_type = tf.uint8
        #converter.inference_output_type = tf.uint8
        converter.experimental_enable_resource_variables = True
        converter.target_spec.supported_types = [tf.int8]

        quantized_model = converter.convert()
        open(name, "wb").write(quantized_model)
    return quantized_model

def explore(model,_dataset,debugger_name=DebuggerName):
    print("\n\n\n\n***************************************************")
    print("explore...\n")
    if (os.path.isfile(DebuggerName)):
        print(f'loading existed file {DebuggerName}')
        with open(debugger_name, 'rb') as f:
            debugger=pickle.load(f)
    elif (os.path.isfile(RESULTS_FILE)):
        print(f'explore not required, existed file {RESULTS_FILE}')
        return 
    else:
        print(f'explore: producing file {DebuggerName}...')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.representative_dataset = rep2(_dataset,N)
        converter.representative_dataset = tf.lite.RepresentativeDataset(rep2(_dataset))
        # my_debug_dataset should have the same format as my_representative_dataset
        #debug_dataset=tf.lite.RepresentativeDataset(rep3(_dataset))
        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=converter, debug_dataset=rep2(_dataset))
        #with open(debugger_name, 'wb') as f:
        #    pickle.dump(debugger, f)

    return debugger

def run_debugger(debugger,res_file):
    print("\n\n\n\n***************************************************")
    print("run debugger...\n")
    if not (os.path.isfile(res_file)):
        print(f'run_debugger: producing file {res_file}')
        debugger.run()
        with open(res_file, 'w') as f:
            debugger.layer_statistics_dump(f)
    else:
        print(f'run_debugger: file is existed; {res_file}')

def Analyze(res_file=RESULTS_FILE_ANALYZED,t=-.33):
    print("\n\n\n\n***************************************************")
    print("Analyze...\n")
    layer_stats = pd.read_csv(res_file)
    layer_stats.head()
    layer_stats['range'] = 255.0 * layer_stats['scale']
    layer_stats['rmse/scale'] = layer_stats.apply(
        lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
    layer_stats[['op_name', 'range', 'rmse/scale']].head()
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(121)
    ax1.bar(np.arange(len(layer_stats)), layer_stats['range'])
    ax1.set_ylabel('range')
    ax2 = plt.subplot(122)
    ax2.bar(np.arange(len(layer_stats)), layer_stats['rmse/scale'])
    ax2.set_ylabel('rmse/scale')
    plt.show()
    #print(layer_stats[layer_stats['rmse/scale'] > t][['op_name', 'range', 'rmse/scale', 'tensor_name']])
    layer_stats.to_csv(res_file,sep=',')

def selective_quantize(model,_dataset,t=0.33, p=0.5):
    print("\n\n\n\n***************************************************")
    print("selective quantization...\n")
    caching=False
    if (os.path.isfile(QSelectiveName)) and caching:
        print(f'selective quantization: loading existed file {QSelectiveName}')
        with open(QSelectiveName, "rb") as f:
            selective_quantized_model=f.read()
    else:
        print(f'selective_quatize: producing file {QSelectiveName}')
        layer_stats = pd.read_csv(RESULTS_FILE)
        suspected_layers = list(layer_stats[layer_stats['rmse/scale'] > t]['tensor_name'])
        nn=len(list(layer_stats['tensor_name']))
        print(f'Number of layers:{nn}, suspected:{len(suspected_layers)}, unquantized first {int(p*nn)} layers')
        suspected_layers.extend(list(layer_stats[:int(p*nn)]['tensor_name']))
        print(len(suspected_layers))
        print(len(list(set(suspected_layers))))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep(_dataset,N)
        debug_options = tf.lite.experimental.QuantizationDebugOptions(denylisted_nodes=suspected_layers)
        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=converter,debug_dataset=rep(_dataset,N),debug_options=debug_options)
        selective_quantized_model = debugger.get_nondebug_quantized_model()
        open(QSelectiveName, "wb").write(selective_quantized_model)
    return selective_quantized_model

def calibrate(model,_dataset,name=CalibratedName):
    print("\n\n\n\n***************************************************")
    print("calibrate...\n")
    if (os.path.isfile(CalibratedName)):
        print(f'calibrate: loading existing file {CalibratedName}')
        with open(name, 'rb') as f:
            calibrated_model = f.read()
    else:
        print(f"calibrate producing file {CalibratedName}")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.representative_dataset = rep2(_dataset)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter._experimental_calibrate_only = True
        converter.inference_input_type = input_dtype
        converter.inference_output_type = output_dtype
        converter.inference_input_shape = input_shape
        converter.inference_output_shape = output_shape
        calibrated_model = converter.convert()
        open(name, "wb").write(calibrated_model)
    return calibrated_model
    



def explore_propogation(calibrated_model,_dataset,debugger_name=DebuggerPropogateName):
    print("\n\n\n\n***************************************************")
    print("explore propogation...\n")
    if (os.path.isfile(DebuggerPropogateName)):
        print(f'explore propogation: loading existed file {DebuggerPropogateName}')
        with open(debugger_name, 'rb') as f:
            debugger=pickle.load(f)
    elif (os.path.isfile(RESULTS_FILE_Propogate)):
        print(f'explore propogate not required, existed file {RESULTS_FILE_Propogate}')
        return
    else:
        print(f"explore_propogation: Producing file {DebuggerPropogateName}")
        # Note that enable_numeric_verify and enable_whole_model_verify are set.
        quantized_model = convert.mlir_quantize(
            calibrated_model,
            enable_numeric_verify=True,
            enable_whole_model_verify=True)
        debugger = tf.lite.experimental.QuantizationDebugger(
            quant_debug_model_content=quantized_model,
            debug_dataset=rep2(_dataset))
        #with open(debugger_name, 'wb') as f:
        #    pickle.dump(debugger, f)
    return debugger

def explore_combinations(calibrated_model,suspected_layers=[],t=0.35,name=UQSelectiveName):
    print("\n\n\n\n***************************************************")
    print("explore combination...\n")
    if (os.path.isfile(name)):
        print(f'explore combinations: loading existed file {name}')
        with open(name, 'rb') as f:
            selective_quantized_model = f.read()
    else:
        print(f"explore_combinations: producing file {name}")
        layer_stats = pd.read_csv(RESULTS_FILE)
        suspected_layers.extend(list(layer_stats[layer_stats['rmse/scale'] > t]['tensor_name']))
        suspected_layers.extend(list(layer_stats[:10]['tensor_name']))
        selective_quantized_model = convert.mlir_quantize(calibrated_model, denylisted_nodes=suspected_layers)
        open(name, "wb").write(selective_quantized_model)
    return selective_quantized_model

def explore_combinations2(calibrated_model,suspected_layers=[],name=UQSelectiveName):
    print("\n\n\n\n***************************************************")
    print("explore combination...\n")
    caching=False
    if (os.path.isfile(name)) and caching:
        print(f'explore combinations: loading existed file {name}')
        with open(name, 'rb') as f:
            selective_quantized_model = f.read()
    else:
        print(f"explore_combinations: producing file {name}")
        selective_quantized_model = convert.mlir_quantize(calibrated_model, denylisted_nodes=suspected_layers)
        with open(name, "wb") as f:
            f.write(selective_quantized_model)
    return 

'''
def amend_input(m=CalibratedName):
    interpreter = tf.lite.Interpreter(model_path=m)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_shape[1] = 608
    input_shape[2] = 608
    interpreter.resize_tensor_input(0, input_shape)
    interpreter.allocate_tensors()
    print(interpreter.get_input_details())
    converter = tf.lite.TFLiteConverter.from_interpreter(interpreter)
    #converter.allow_custom_ops = True  # If you have any custom ops in your model
    tflite_model = converter.convert()
    with open('modified_model.tflite', 'wb') as f:
        f.write(tflite_model)

'''


# -


def initialize():
    global model,dataset,calibrated_model
    model=load_model(m=ModelName)
    model.summary()
    dataset=load_dataset()
    # Print the first 5 images in the dataset
    for image in dataset.take(5):
        print(image.shape)
    #global calibrated_model
    calibrated_model=calibrate(model,dataset)


# +

#import imp
#imp.reload(tt)
def evaluate(model_name, pkl_name):
    #cmd='cd ../Evaluation; python ../Evaluation/eval_multiThread2.py '+args
    #os.system(cmd)
    mAP,APs=tt.main(["--num_threads",'64',"--model_path",model_name,"--pkl_name",pkl_name])
    return mAP,APs


# -

def generate_indexes(start_conv=-1,end_conv=-1):
    ####
    layer_stats = pd.read_csv(RESULTS_FILE)
    all_layers=list(layer_stats[:]['tensor_name'])
    conv_layers=[]
    for i,layer in enumerate(all_layers):
        if 'conv' in layer or 'StatefulPartitionedCall' in layer:
            conv_layers.append(layer)
            
    _n=len(conv_layers)
    #cases=[ [ conv_layers[start:end+1] for end in range(start,_n) ] for start in range(0,_n) ]
    #n_cases = [len(case) for case in cases ]
    #N_cases = sum(n_cases)
    cases=[  list(range(start,end+1))  for start in range(0,_n) for end in range(start,_n)]
    N_cases = len(cases)
    print(f'Total layers:{len(all_layers)}  Convs:{len(conv_layers)}  number of cases:{N_cases}')
    #flatted_cases=[c for case in cases for c in case]
    last_conv_indx=len(conv_layers)-1
    for i,case in enumerate(cases):
        start_conv=case[0]
        end_conv=case[-1]
        start_index=all_layers.index(conv_layers[start_conv])
        if end_conv==last_conv_indx:
            end_index=len(all_layers)-1
        else:
            end_index=all_layers.index(conv_layers[end_conv+1])
        suspend=all_layers[0:start_index]+all_layers[end_index:]
        #kk=list(set(all_layers)-set(suspend))
        #print(f'quantizing conv layers from {start_conv} to {end_conv}')
        #print(f'index {start_index} to {end_index}')
        #print(all_layers[start_index:end_index])
        #ttt=explore_combinations2(calibrated_model,suspected_layers=ss,name='2-3.tflite')'''
        ##next_start_conv=cases[i+1]
        ##next_end_conv=cases
        yield i,N_cases,start_conv,end_conv,start_index,end_index,suspend



# +
        
def run():
    output=os.getcwd()+"/cases/"
    os.makedirs(output, exist_ok=True)
    i=0
    pklDatafile="DataResults.pkl"
    dffile="df.csv"
    if os.path.isfile(dffile):
        df=pd.read_csv(dffile,index_col=0)
        #i=df.iloc[-1][0]+1
        i=len(df)
        print(f'Continue {dffile} from index {i}')
    else:
        response=input("Do you want to reset df.csv? yes/*   ")
        if response=="yes" or response=="Yes":
            df = pd.DataFrame(columns=["name","mAP"])
        else:
            return
        
    if os.path.isfile(pklDatafile):
        with open(pklDatafile,'rb') as f:
            Data=pickle.load(f)
        print(f'{pklDatafile} is loaded')
    else:
        response=input("Do you want to reset DataResults.pkl? yes/*   ")
        if response=="yes" or response=="Yes":
            Data=[]
        else:
            return
        
    for c in generate_indexes():
        '''if c[0]<i:
            continue'''
        #input(f'i is {i}')
        print("\n\n\n*****************\n\n\n")
        print(f'Case:{c[0]}/{c[1]}')
        print(f'quantizing conv layers from {c[2]} to {c[3]}')
        print(f'index {c[4]} to {c[5]}')  
        _name=f'{c[2]}-{c[3]}'
        if df[df['name']==_name].shape[0]:
            print("Already evaluated...")
            continue
        
        m_name=output+_name+'.tflite'
        p_name=_name+'.pkl'
        
        start_time=time.time()
        explore_combinations2(calibrated_model,suspected_layers=c[-1],name=m_name)
        end_time=time.time()
        print(f"{m_name} Quantization finished time: {end_time-start_time}")
        
        mAP,APs=evaluate(model_name=m_name,pkl_name=p_name)
        end_time=time.time()
        print(f"{m_name} Evaluation finished time: {end_time-start_time}")
        
        os.remove(m_name)
        df.loc[c[0]]=[_name,mAP]
        dct={"i":c[0],"start_conv":c[2], "end_conv":c[3], "start_index":c[4], "end_index":c[5],"mAP":mAP, "APs":APs}
        Data.append(dct)
        if c[0]%5==0 or True:
            df.to_csv(dffile)
            with open(pklDatafile,'wb') as f:
                pickle.dump(Data,f)

                


# # +
def generate_indexes_2(start_conv=-1,end_conv=-1):
    ####
    layer_stats = pd.read_csv(RESULTS_FILE)
    all_layers=list(layer_stats[:]['tensor_name'])
    conv_layers=[]
    #print(all_layers)
    for i,layer in enumerate(all_layers):
        if 'conv' in layer or 'StatefulPartitionedCall' in layer:
            conv_layers.append(layer)
            
    _n=len(conv_layers)
    #cases=[ [ conv_layers[start:end+1] for end in range(start,_n) ] for start in range(0,_n) ]
    #n_cases = [len(case) for case in cases ]
    #N_cases = sum(n_cases)
    #cases = list(itertools.combinations(conv_layers, 2))
    _convs=list(range(len(conv_layers)))
    cases=list(itertools.combinations(_convs, 1))
    cases+=list(itertools.combinations(_convs, 2))
    N_cases = len(cases)
    print(f'Total layers:{len(all_layers)}  Convs:{len(conv_layers)}  number of cases:{N_cases}')
    #flatted_cases=[c for case in cases for c in case]
    last_conv_indx=len(conv_layers)-1
    last_conv=conv_layers[-1]
    for i,case in enumerate(cases):
        print(f'case:\n{case}')
        suspend=all_layers[:]
        quant=[]
        for layer in case:           
            start_index=all_layers.index(conv_layers[layer])
            if layer==last_conv_indx:
                end_index=len(all_layers)-1
            else:
                end_index=all_layers.index(conv_layers[layer+1])
            block=[all_layers[j] for j in range(start_index,end_index)]
            quant+=block
        print(quant)         
        suspend=[elem for elem in all_layers if elem not in quant]
        print(len(quant),len(suspend),len(all_layers))
        
        
        yield i,N_cases,tuple(case),suspend


# + endofcell="------"
# # +
def extract_one_two_from_consequtives():
    m=pd.read_csv('df.csv',index_col=0)
    m['layers'] = m['name'].str.split('-').apply(lambda x: tuple(range(int(x[0]), int(x[1])+1)))
    #pd.set_option('display.max_rows',3000)
    k=1
    m_1=m[m['layers'].apply(len) == 1]
    m_2=m[m['layers'].apply(len) == 2]
    #m_filtered = m[m['layers'].apply(len).isin([1,2])]
    ms=pd.concat([m_1,m_2],ignore_index=True)
    ms.to_csv("extracted_df.csv",index=False)
    return ms
#extract_one_two_from_consequtives()


# +
#t=extract_one_two_from_consequtives()
#t[t['layers'].astype(str)=='(0,)']

# + endofcell="-------"
def run_2():
    output=os.getcwd()+"/cases/"
    os.makedirs(output, exist_ok=True)
    dffile="df2.csv"
    if os.path.isfile(dffile):
        df=pd.read_csv(dffile,index_col=0)
        #i=df.iloc[-1][0]+1
        i=len(df)
        print(f'Continue {dffile} from index {i}')
    else:
        response=input("Do you want to reset df.csv? yes/*   ")
        if response=="yes" or response=="Yes":
            df = pd.DataFrame(columns=["name","mAP"])
        else:
            return
        
    
    extracted_df=extract_one_two_from_consequtives()
    
    for c in generate_indexes_2():
        
        print("\n\n\n*****************\n\n\n")
        print(f'Case:{c[0]}/{c[1]}')
        print(f'quantizing conv layers {c[2]}')
        _name=f'{c[2]}'
        if df[df['name']==_name].shape[0]:
            print("Already evaluated...")
            continue
            
        
        if extracted_df[extracted_df['layers'].astype(str)==_name].shape[0]:
            mAP=extracted_df[extracted_df['layers'].astype(str)==_name]['mAP'].iloc[0]
            df.loc[c[0]]=[_name,mAP]
            df.to_csv(dffile)
            print('extract')
            continue
        
        
        m_name=output+_name+'.tflite'
        p_name=_name+'.pkl'
        
        start_time=time.time()
        explore_combinations2(calibrated_model,suspected_layers=c[-1],name=m_name)
        end_time=time.time()
        print(f"{m_name} Quantization finished time: {end_time-start_time}")
        
        mAP,APs=evaluate(model_name=m_name,pkl_name=p_name)
        end_time=time.time()
        print(f"{m_name} Evaluation finished time: {end_time-start_time}")
        
        os.remove(m_name)
        df.loc[c[0]]=[_name,mAP]
        
        if c[0]%5==0 or True:
            df.to_csv(dffile)
            

# # # +
# # # # +
#evaluate([])
### run is for consequtive layer quantizations 
### and run_2 is for select two layer quantization
if __name__ == "__main__":
    initialize()
    if os.path.isfile(RESULTS_FILE):
        run_2()
    else:
        quantized_model=quantize(model,dataset)
        debugger=explore(model,dataset)
        run_debugger(debugger,RESULTS_FILE)
        run_2()


# # # + endofcell="-----"
# # # # # +

# # # # + endofcell="----"
# # # # # # +
def _run2():
    ouput=os.getcwd()+"/cases/"
    os.makedirs(output, exist_ok=True)
    
    
    layer_stats = pd.read_csv(RESULTS_FILE)
    all_layers=list(layer_stats[:]['tensor_name'])
    conv_layers=[]
    for i,layer in enumerate(all_layers):
        if 'conv' in layer or 'StatefulPartitionedCall' in layer:
            conv_layers.append(layer)
            
    _n=len(conv_layers)
    #cases=[ [ conv_layers[start:end+1] for end in range(start,_n) ] for start in range(0,_n) ]
    cases=[  list(range(start,end+1))  for start in range(0,_n) for end in range(start,_n)]
    n_cases = [len(case) for case in cases ]
    N_cases = sum(n_cases)
    print(f'Total layers:{len(all_layers)}  Convs:{len(conv_layers)}  number of cases:{N_cases}')
    
    def data_case(i):
        Data={}
        start_conv=cases[i][0]
        end_conv=cases[i][-1]
        start_index=all_layers.index(conv_layers[start_conv])
        if end_conv==len(conv_layers)-1:
            end_index=len(all_layers)-1
        else:
            end_index=all_layers.index(conv_layers[end_conv+1])
        suspend=all_layers[0:start_index]+all_layers[end_index:]
        _name=f'{start_conv}-{end_conv}'
        
        Data['start_conv']=start_conv
        Data['end_conv']=end_conv
        Data['suspend']=suspend
        Data['_name']=_name
        return Data
        
    Data1=data_case(0)
    thread = threading.Thread(target=explore_combinations2,args=(calibrated_model,suspected_layers:=c[-1],name:=m_name))
    thread.start()
    for j in range(1,len(cases)):
        
        #kk=list(set(all_layers)-set(suspend))
        print("\n\n\n*****************")
        print(f'quantizing conv layers from {start_conv} to {end_conv}')
        print(f'index {start_index} to {end_index}')
        print(all_layers[start_index:end_index])
        print(f'Case:{i}/{N_cases}')
        _name=f'{start_conv}-{end_conv}'
        m_name=output+_name+'.tflite'
        p_name=_name+'.pkl'
        if i==0:
            thread = threading.Thread(target=explore_combinations2,args=(calibrated_model,suspected_layers:=c[-1],name:=m_name))
            thread.start()
            thread.join()
            
        next_start_conv=cases[i+1][0]
        next_end_conv=cases[i+1][-1]
        next_start_index=all_layers.index(conv_layers[next_start_conv])
        next_end_index=all_layers.index(conv_layers[next_end_conv+1])
        next_suspend=all_layers[0:start_index]+all_layers[end_index:]
        
    for c in generate_indexes():
        threads=[]
        start_time=time.time()
        print("\n\n\n*****************\n\n\n")
        print(f'Case:{c[0]}/{c[1]}')
        _name=f'{c[2]}-{c[3]}'
        m_name=output+_name+'.tflite'
        p_name=_name+'.pkl'
        thread = threading.Thread(target=explore_combinations2,args=(calibrated_model,suspected_layers:=c[-1],name:=m_name))
        end_time=time.time()
        print(f"{mname} Quantization finished time: {end_time-start_time}")
        ## Trigger Evaluation
        thread = threading.Thread(target=evaluate, args=[model_name:=m_name,pkl_name:=p_name])
        threads.append(thread)

    for thread in threads:
        thread.join()



def generate_cases_keras():
    all_layers=[l.name for l in model.layers]
    conv_layers=[name for (indx,name) in enumerate(all_layers) if 'conv' in name]
    _n=len(conv_layers)
    cases=[ [ conv_layers[start:end+1] for end in range(start,_n) ] for start in range(0,_n) ]
    n_cases = [len(case) for case in cases ]
    N_cases = sum(n_cases)
    flatted_cases=[c for case in cases for c in case]
    #test=flatted_cases[4]
    #t=explore_combinations2(calibrated_model,suspected_layers=test,name='ttt.tflite')
    return flatted_cases

def run_keras():
    global calibrated_model
    threads=[]
    for case in flatted_cases:
        start_time=time.time()
        c=f'{case[0]}-{case[-1]}'
        mname=resdir+c+'.tflite'
        print(f"Quantizaing layers {c} --> {mname}")
        explore_combinations2(calibrated_model,suspected_layers=case,name=mname)
        end_time=time.time()
        print(f"{mname} Quantization finished time: {end_time-start_time}")
        thread = threading.Thread(target=evaluate, args=[model_name:=mname,pkl_name:=f'{c}.pkl'])
        threads.append(thread)
        input("one_run...")

    for thread in threads:
        thread.join()


# # # # # + endofcell="--"
# -

'''
sq=selective_quantize(model,dataset,t=0.31,p=0.3)
calibrated_model=calibrate(model,dataset)
quantized_model=quantize(model,dataset)
debugger=explore(model,dataset)
run_debugger(debugger,RESULTS_FILE)
Analyze(RESULTS_FILE_ANALYZED)
selective_quantized=selective_quantize(model,dataset)
calibrated_model=calibrate(model,dataset)
debugger=explore_propogation(calibrated_model,dataset)
run_debugger(debugger,RESULTS_FILE_Propogate)
Analyze(RESULTS_FILE_Propogate_ANALYZED)
calibrated_model=calibrate(model,dataset)
selective_unquantized=explore_combinations(calibrated_model)'''


# # # # # # # # # %%timeit -n 1 -r 1
def ttt():
    QuantizedName='Yolo_files/1/YoloV3_quztized.tflite'
    model = interpreter_wrapper.Interpreter(model_path=QuantizedName)
    model.allocate_tensors()
    model_format = 'TFLITE'
    model_input_shape=(608,608)
    #Ehsan input shape correctness
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    input_shape = input_details[0]['shape']

    n=100
    input_shape[0]=n
    input_shape[1] = model_input_shape[0]
    input_shape[2] = model_input_shape[1]
    model.resize_tensor_input(0, input_shape)
    model.allocate_tensors()
    print(input_shape)
    input_shape=[n,608,608,3]
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    # Set the input tensor to the interpreter
    model.set_tensor(input_details[0]['index'], input_data)

    # Run the model on the GPU
    with tf.device('/gpu:1'):
        model.invoke()

    # Get the output tensor from the interpreter
    output_data = model.get_tensor(output_details[0]['index'])

    output_data
# ---
# --
# ----
# -----
# ------
# -------


a=list(range(10))
import itertools
cases=list(itertools.combinations(a, 1))
cases+=list(itertools.combinations(a, 2))
cases


