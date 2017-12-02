import tensorflow as tf
slim = tf.contrib.slim

from PIL import Image
from inception_resnet_v2 import *
import numpy as np
import os

class Predictor:
	def __init__(self):
		self.name_list = []
		self.code_list = []
		#use unique tf.Graph() to avoid variable name conflict in multi-models.
		self.graph = tf.Graph()

	def __del__(self):
		self.sess.close()

	def load_ckpt(self, ckpt_dir, ckpt_name, label_file='labels.txt'):
		self.ckpt_name = ckpt_name
		#label_file is made up of lines containing name:code
		with open(os.path.join(ckpt_dir,label_file),'r') as f:
			for line in f:
				_type = line.split(":")
				self.name_list.append(_type[0].strip())
				self.code_list.append(_type[1].strip())
			self.num_classes = len(self.name_list)

		self.sess = tf.Session(graph = self.g)
		with self.g.as_default():
			scaled_input_tensor1 = tf.scalar_mul((1.0/255), self.input_tensor)
			scaled_input_tensor2 = tf.subtract(scaled_input_tensor1, 0.5)
			scaled_input_tensor = tf.multiply(scaled_input_tensor2, 2.0)

			arg_scope = inception_resnet_v2_arg_scope()
			with slim.arg_scope(arg_scope) as scope:
				self.logits, self.end_points = inception_resnet_v2(scaled_input_tensor, num_classes=self.num_classes, is_training=False)
			saver = tf.train.Saver()
			saver.restore(self.sess, os.path.join(ckpt_dir, ckpt_name))

	def get_predict(self,img):
		img = img.resize((299,299))
		img = np.array(img)
		img = img.reshape(-1,299,299,3)
		with self.g.as_default():
			predict_values, logit_values = self.sess.run([self.end_points['Predictions'], self.logits], feed_dict={self.input_tensor: img})
		return predict_values[0]

#argv[1]:ckpt_dir
#argv[2]:ckpt_file
#argv[3]:img_file
if __name__ == "__main__":
	import sys
	if len(sys.argv) < 3:
		print "python predict.py ckpt_file img_file"
		sys.exit(0)

	from time import clock
	predictor = Predictor()
	start = clock()
	predictor.load_ckpt(sys.argv[1], sys.argv[2])
	end = clock()
	print "load time =",end-start
	start = clock()
	pred = predictor.get_predict(Image.open(sys.argv[3]))
	print pred
	end = clock()
	print "predict time =",end-start
