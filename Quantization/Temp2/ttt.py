import tensorflow as tf

# Load the original Keras model
model = tf.keras.models.load_model('Yolov3.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_session(tf.keras.backend.get_session(), input_tensors=[model.input], output_tensors=[model.output])

# Specify the layers to quantize
converter.inference_input_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
for input_array in input_arrays:
    converter.quantized_input_stats[input_array] = (0., 1.)
converter.default_ranges_stats = (0, 6)

# Convert and save the quantized model
tflite_model = converter.convert()
open("model_quant.tflite", "wb").write(tflite_model)

