import tensorflow as tf
slim = tf.contrib.slim

from PIL import Image
from inception_resnet_v2 import *
import numpy as np

class Predictor:
	def __init__(self):
		#self.sess = None
		self.input_tensor = tf.placeholder(tf.float32, shape=(None,299,299,3))
		#self.logits = None
		#self.end_points = None

	def load_ckpt(self, ckpt_file):
		scaled_input_tensor1 = tf.scalar_mul((1.0/255), self.input_tensor)
		scaled_input_tensor2 = tf.subtract(scaled_input_tensor1, 0.5)
		scaled_input_tensor = tf.multiply(scaled_input_tensor2, 2.0)

		arg_scope = inception_resnet_v2_arg_scope()
		with slim.arg_scope(arg_scope) as scope:
			self.logits, self.end_points = inception_resnet_v2(scaled_input_tensor, num_classes=100, is_training=False)#, reuse=False, scope="InceptionResnetV2")
		self.sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(self.sess, ckpt_file)

	def get_predict(self,img):
		img = img.resize((299,299))
		img = np.array(img)
		img = img.reshape(-1,299,299,3)
		predict_values, logit_values = self.sess.run([self.end_points['Predictions'], self.logits], feed_dict={self.input_tensor: img})
		return predict_values[0]

#argv[1]:ckpt_file
#argv[2]:img_file
if __name__ == "__main__":
	import sys
	if len(sys.argv) < 3:
		print "python predict.py ckpt_file img_file"
		sys.exit(0)

	from time import clock
	predictor = Predictor()
	start = clock()
	predictor.load_ckpt(sys.argv[1])
	end = clock()
	print "load time =",end-start
	start = clock()
	pred = predictor.get_predict(Image.open(sys.argv[2]))
	print pred
	end = clock()
	print "predict time =",end-start
