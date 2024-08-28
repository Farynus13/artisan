import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.models.load_model('model.keras')
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, "model.onnx")