import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("blazeface_tf")
tflite_model = converter.convert()

with open("blazeface.tflite", "wb") as f:
    f.write(tflite_model)
