from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import time
import os
from collections import deque # 双端队列


BATCH = 50 # 随机抽样的样本数


def DataPre():
	
	#判断是否已经处理过数据了
	if os.path.exists('DataPre.csv'):
		print("Data has been handled")
		return
	else:
		print("Data Pre")
	
	df= pd.read_csv("EasyBoston.csv",sep=",")
	df_norm = (df - df.min()) / (df.max() - df.min())
	df_norm.to_csv('InputForTrainAndTestNoY.csv',index=False)
	
	

class DNN:

	def __init__(self):
	
		self.memory = deque()
		
		self.createNetwork()
		# 保存和加载网络模型
		self.saver = tf.train.Saver()
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
		#如果检查点存在就载入已经有的模型
		checkpoint = tf.train.get_checkpoint_state("Model/")
		if checkpoint:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")
	#权重
	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.01) 
		return tf.Variable(initial)

    #偏置
	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)
	
	
	def createNetwork(self):
		# 输入层 - 隐藏层
		W_fc1 = self.weight_variable([13, 40])
		b_fc1 = self.bias_variable([40])
		# 隐藏层1 - 隐藏层2
		W_fc3 = self.weight_variable([40, 1])
		b_fc3 = self.bias_variable([1])


		self.inputState = tf.placeholder("float", [None, 13])
		
		# 第一层隐藏层
		h_fc1 = tf.nn.relu(tf.matmul(self.inputState, W_fc1) + b_fc1)
		#输出层
		self.y_conv = tf.matmul(h_fc1, W_fc3) + b_fc3
		
		# 损失函数
		self.y_ = tf.placeholder("float", shape=[None,1])
		self.temp = tf.square(self.y_ - self.y_conv)
		self.mse = tf.reduce_mean(self.temp)
		self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.mse)
		
	def	trainNetwork(self,x,y):
		print("START Train")
		c = 0
		while(c < 10000):
			
			# 梯度下降
			minibatch = random.sample(self.memory, BATCH)
			state_batch  = [d[0] for d in minibatch]
			y_batch = [d[1] for d in minibatch]
			state_batch = np.reshape(state_batch, (-1, 13))
			y_batch = np.reshape(y_batch, (-1, 1))
			# perform gradient step
			self.train_step.run(feed_dict = {self.inputState : state_batch, self.y_ : y_batch})
			
			#每次训练后随机抽样，检查模型预测是否符合条件，累计超过10000次则认为模型已经训练好了
			#用其他的方式容易训练不足
			j = random.randint(0,505)
			input = np.matrix(x[j])
			y_ = self.y_conv.eval(feed_dict={self.inputState:input})
			t = y_[0][0] - y[j][0]
			if(abs(t) < 0.001):
				c = c + 1
				print((str(y[j][0])) + " " + (str(y_[0][0])))
			
		
		#保存模型
		self.saver.save(self.sess,'DNN')
			
		
	def handleData(self, matrix, label):
		for i in range(matrix.shape[0]):
			input = np.matrix(matrix[i])
			y  = np.matrix(label[i])
			self.memory.append((input, y))
		print("样本池：" + str(len(self.memory)))
	
	def useDNN(self, matrix):
		y_conv = self.y_conv.eval(feed_dict={self.inputState:matrix})
		return y_conv

def train():

	dnn = DNN()
	data = pd.read_table("InputForTrainAndTestNoY.csv",sep=",")
	x = data.ix[:,['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']].as_matrix()
	y = data.ix[:,['MEDV']].as_matrix()
	#print(x)
	dnn.handleData(x,y)
	dnn.trainNetwork(x,y)
	

	

def test():
	dnn = DNN()
	print('Test')
	print('Test')
	print('Test')

		

	data = pd.read_table("InputForTrainAndTestNoY.csv",sep=",")
	x = data.ix[:,['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']].as_matrix()
	y = data.ix[:,['MEDV']].as_matrix()
	
	ava = 0
	for i in range(200):
		j = random.randint(0,505)
		input = np.matrix(x[j])
		y_ = dnn.useDNN(input)
		y_ = y_[0][0]
		y_j = y[j][0]
		
		ava = ava + abs(y_ - y_j)
		print((str(y_j)) + " " + (str(y_)))
	ava = ava / 200
	print('平均误差：' + str(ava))
	

def main():
	DataPre()
	#train()
	test()


if __name__ == "__main__":
	main()