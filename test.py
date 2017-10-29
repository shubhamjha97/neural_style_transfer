import vgg16
import cv2
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #ignore tf warnings
import time

def content_loss(a,b):
	return tf.div(tf.sqrt(tf.reduce_mean(tf.square(a - b))), 2)

def gram(a):
	sh=a.get_shape()
	n_channels=sh[3]
	matrix = tf.reshape(a, shape=[-1, 128])
	gram= tf.matmul(tf.transpose(matrix), matrix)
	#print('Gram shape:', gram.get_shape())
	return tf.div(gram, 4*(112*112)**2*(128)**2)

def style_loss(c,d):
	#c=c
	losses=[]
	for c_, d_ in zip(c,d):
		d__=tf.constant(d_)
		#print('d shape:',d__.get_shape(), d__.dtype)
		#print('c shape:',c_.get_shape(), c_.dtype)
		gram_d=gram(d__)
		gram_c=gram(c_)
		losses.append(tf.sqrt(tf.reduce_mean(tf.square(gram_c - gram_d))))
	return tf.reduce_mean(losses)

def create_denoise_loss(model):
    loss = tf.reduce_mean(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_mean(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
    return loss

def train():
	train_start_time=time.time()
	style_img=cv2.imread('comic.jpg')
	content_img=cv2.imread('putin.jpg')
	#rand_img=np.random.randint(low=0, high=256, size=[512, 512, 3])
	rand_img=np.random.rand(512,512,3)+128
	rand_img=rand_img.astype(float)
	net=vgg16.VGG16()
	#print(net.layer_names)
	content_layer=[4]
	style_layer=range(13)
	a=net.get_layer_tensors(layer_ids=content_layer)
	#print(a)
	feed_dict_content = net.create_feed_dict(image=content_img)
	feed_dict_style = net.create_feed_dict(image=style_img)
	sess=tf.Session(graph=net.graph)
	temp=sess.run(a, feed_dict=feed_dict_content)
	print('Temp len:', temp[0].shape)
	with net.graph.as_default():
		b=tf.constant(temp[0])
		print(b.get_shape())
		con_l=content_loss(a[0],b)
		c=net.get_layer_tensors(layer_ids=style_layer)
		temp_1=sess.run(c, feed_dict=feed_dict_style)
		#d=tf.constant(temp_1)
		style_l=style_loss(c, temp_1)
		denoise_l=create_denoise_loss(net)
		#total_loss=style_l+0.0001*con_l+0.00001*denoise_l
		total_loss=style_l+0.0003*con_l+0.0001*denoise_l
		num_iter=1000
		grad=tf.gradients(total_loss, net.input)
		grad_mag=tf.reduce_mean(tf.reshape(grad, [-1]))
		lr=10000000
		for i in range(num_iter):
			start_time=time.time()
			feed_dict_rand = net.create_feed_dict(image=rand_img)
			l, grads, g_m=sess.run([total_loss, grad, grad_mag], feed_dict=feed_dict_rand)
			#print('grads shape:', len(grads))
			rand_img-=lr*grads[0].reshape([512,512,3])
			print('Step:', i, 'Loss:', l, 'Gradients magnitude:', g_m, 'LR:', lr, 'Time per iter:', time.time()-start_time, 
				'Total running time:', (time.time()-train_start_time)/60.0)
			if i%5==0:
				cv2.imwrite('sequence/'+str(i)+'temp.jpg', rand_img.clip(0,255))
				print('Writing temp to file....')
			if lr>1000000:
				#print('LR update:', lr)	
				lr=0.99*lr

	cv2.imwrite('final.jpg', rand_img.clip(0,255))
	print('Total time taken:', time.time()-train_start_time)

train()