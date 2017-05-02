#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:33:29 2017

The goal of this script is to study the gradient values

@author: nicolas
"""
import scipy
import time
import numpy as np
import tensorflow as tf
import Style_Transfer as st
import pickle
from Arg_Parser import get_parser_args 
import seaborn as sns
import matplotlib.pyplot as plt

content_layers = [('conv4_2',1.)]
#style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]
style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.)]
#style_layers = [('conv1_1',1.)]

def sum_content_losses(sess, net, dict_features_repr,M_dict):
	"""
	Compute the content term of the loss function
	Input : 
	- the tensforflow session sess
	- the vgg19 net
	- the dictionnary of the content image representation thanks to the net
	"""
	length_content_layers = float(len(content_layers))
	weight_help_convergence = 10**(5) 
	content_loss = 0
	for layer, weight in content_layers:
		M = M_dict[layer[:5]]
		# If we add the M normalization we get a content loss smaller and so on a bigger emphasis on the style !
		P = tf.constant(dict_features_repr[layer])
		F = net[layer]
		content_loss +=  tf.nn.l2_loss(tf.subtract(P,F))* (weight*weight_help_convergence/(length_content_layers*(M**2)))
	return(content_loss)

def grad_loss_content_norm(sess,net,dict_features_repr,M_dict):
	length_content_layers = float(len(content_layers))
	grad_content_loss = 0
	maxlist = []
	minlist = []
	for layer, weight in content_layers:
		M = M_dict[layer[:5]]
		P = tf.constant(dict_features_repr[layer])
		F = net[layer]
		grad = tf.subtract(P,F)* (2*weight/(length_content_layers*M))
		maxlist.append(tf.reduce_max(tf.abs(grad)))
		minlist.append(tf.reduce_min(tf.abs(grad)))
		grad_content_loss +=  tf.nn.l2_loss(grad)
	maximum = tf.reduce_max(maxlist)
	minimum = tf.reduce_min(minlist)
	return(grad_content_loss,maximum,minimum)

def grad_loss_style_norm(sess, net, dict_gram,M_dict):
	# TODO : be able to choose more quickly the different parameters
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	# Info for the vgg19
	length_style_layers = float(len(style_layers)) 
	# Because the function is pretty flat 
	grad_total_style_loss = 0
	maxlist = []
	minlist = []
	for layer, weight in style_layers:
		# For one layer
		N = style_layers_size[layer[:5]]
		A = dict_gram[layer]
		A = tf.constant(A)
		# Get the value of this layer with the generated image
		M = M_dict[layer[:5]]
		x = net[layer]
		G = st.gram_matrix(x,N,M)
		x = tf.transpose(x,(0,3,1,2))
		F = tf.reshape(x,[tf.to_int32(N),tf.to_int32(M)])
		M =  tf.to_float(M)
		grad = tf.matmul(tf.transpose(F),tf.subtract(G,A))*weight   / ((N**4)*length_style_layers) 
		maxlist.append(tf.reduce_max(tf.abs(grad)))
		minlist.append(tf.reduce_min(tf.abs(grad)))
		grad_style_loss = tf.nn.l2_loss(grad)  # output = sum(t ** 2) / 2
		grad_total_style_loss += grad_style_loss # 
	maximum = tf.reduce_max(maxlist)
	minimum = tf.reduce_min(minlist)
	return(grad_total_style_loss,maximum,minimum)

def sum_style_losses(sess, net, dict_gram,M_dict):
	"""
	Compute the style term of the loss function 
	Input : 
	- the tensforflow session sess
	- the vgg19 net
	- the dictionnary of Gram Matrices
	- the dictionnary of the size of the image content through the net
	"""
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	# Info for the vgg19
	length_style_layers = float(len(style_layers))
	weight_help_convergence = 2*10**(9) # This wight come from a paper of Gatys
	# Because the function is pretty flat 
	total_style_loss = 0
	for layer, weight in style_layers:
		# For one layer
		N = style_layers_size[layer[:5]]
		A = dict_gram[layer]
		A = tf.constant(A)
		# Get the value of this layer with the generated image
		M = M_dict[layer[:5]]
		x = net[layer]
		G = st.gram_matrix(x,N,M)
		style_loss = tf.nn.l2_loss(tf.subtract(G,A))  # output = sum(t ** 2) / 2
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
		total_style_loss += style_loss
	return(total_style_loss)

def grad_computation(args):
	sns.set()
	output_image_path = args.img_folder + args.output_img_name +args.img_ext
	image_content_path = args.img_folder + args.content_img_name +args.img_ext
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	image_content = st.preprocess(scipy.misc.imread(image_content_path).astype('float32')) # Float between 0 and 255
	image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	_,image_h, image_w, number_of_channels = image_content.shape 
	_,image_h_art, image_w_art, _ = image_style.shape 
	M_dict = st.get_M_dict(image_h,image_w)
	#print("Content")
	#plt.figure()
	#plt.imshow(postprocess(image_content))
	#plt.show()
	#print("Style")
	#plt.figure()
	#plt.imshow(postprocess(image_style))
	#plt.show()
	# TODO add something that reshape the image 
	# TODO : be able to have two different size for the image
	t1 = time.time()

	
	vgg_layers = st.get_vgg_layers()
	
	data_style_path = args.data_folder + "gram_"+args.style_img_name+"_"+str(image_h_art)+"_"+str(image_w_art)+".pkl"
	try:
		dict_gram = pickle.load(open(data_style_path, 'rb'))
	except(FileNotFoundError):
		if(args.verbose): print("The Gram Matrices doesn't exist, we will generate them.")
		dict_gram = st.get_Gram_matrix(vgg_layers,image_style)
		with open(data_style_path, 'wb') as output_gram_pkl:
			pickle.dump(dict_gram,output_gram_pkl)
		if(args.verbose): print("Pickle dumped")

	data_content_path = args.data_folder +args.content_img_name+"_"+str(image_h)+"_"+str(image_w)+".pkl"
	try:
		dict_features_repr = pickle.load(open(data_content_path, 'rb'))
	except(FileNotFoundError):
		if(args.verbose): print("The dictionnary of features representation of content image doesn't exist, we will generate it.")
		dict_features_repr = st.get_features_repr(vgg_layers,image_content)
		with open(data_content_path, 'wb') as output_content_pkl:
			pickle.dump(dict_features_repr,output_content_pkl)
		if(args.verbose): print("Pickle dumped")


	net = st.net_preloaded(vgg_layers, image_content) # The output image as the same size as the content one
	t2 = time.time()
	print("net loaded and gram computation after ",t2-t1," s")

	try:
		sess = tf.Session()
		
		if(not(args.start_from_noise)):
			try:
				init_img = st.preprocess(scipy.misc.imread(output_image_path).astype('float32'))
			except(FileNotFoundError):
				if(args.verbose): print("Former image not found, use of white noise mixed with the content image as initialization image")
				# White noise that we use at the beginning of the optimization
				init_img = st.get_init_noise_img(image_content,args.init_noise_ratio)
		else:
			init_img = st.get_init_noise_img(image_content,args.init_noise_ratio)

		#
		# TODO add a plot mode ! 
		#noise_imgS = postprocess(noise_img)
		#plt.figure()
		#plt.imshow(noise_imgS)
		#plt.show()
		
		# Propose different way to compute the lossses 
		style_loss = sum_style_losses(sess,net,dict_gram,M_dict)
		content_loss = args.content_strengh * sum_content_losses(sess, net, dict_features_repr,M_dict) # alpha/Beta ratio 
		loss_total =  content_loss + style_loss
		grad_style_loss = grad_loss_style_norm(sess,net,dict_gram,M_dict)
		grad_content_loss = grad_loss_content_norm(sess, net, dict_features_repr,M_dict)
		
		style_loss_tab = np.zeros((args.max_iter,1))
		content_loss_tab = np.zeros((args.max_iter,1))
		loss_total_tab =  np.zeros((args.max_iter,1))
		grad_style_loss_tab = np.zeros((args.max_iter,1))
		grad_content_loss_tab = np.zeros((args.max_iter,1))
		grad_style_loss_min = np.zeros((args.max_iter,1))
		grad_content_loss_min = np.zeros((args.max_iter,1))
		grad_style_loss_max = np.zeros((args.max_iter,1))
		grad_content_loss_max = np.zeros((args.max_iter,1))
		
		
		print("init loss total")

		optimizer = tf.train.AdamOptimizer(args.learning_rate) # Gradient Descent
		# TODO function in order to use different optimization function
		train = optimizer.minimize(loss_total)

		sess.run(tf.global_variables_initializer())
		sess.run(net['input'].assign(init_img)) # This line must be after variables initialization ! 
		t3 = time.time()
		print("sess Adam initialized after ",t3-t2," s")
		# turn on interactive mode
		print("loss before optimization")
		st.print_loss(sess,loss_total,content_loss,style_loss)
		for i in range(args.max_iter):
			if(i%args.print_iter==0):
				print(i)
			sess.run(train)
			style_loss_tab[i] = sess.run(style_loss)
			content_loss_tab[i] = sess.run(content_loss)
			loss_total_tab[i] = sess.run(loss_total)
			grad_norm_l2,maximum,minimum = sess.run(grad_style_loss)
			grad_style_loss_min[i] = minimum
			grad_style_loss_max[i] = maximum
			grad_style_loss_tab[i] = grad_norm_l2
			grad_content_loss_norml2,maximum,minimum = sess.run(grad_content_loss)
			grad_content_loss_tab[i] = grad_content_loss_norml2
			grad_content_loss_min[i] = minimum
			grad_content_loss_max[i] = maximum
		
	except:
		print("Error")
		raise 
	finally:
		print("Close Sess")
		result_img = sess.run(net['input'])
		result_img_postproc = st.postprocess(result_img)
		scipy.misc.toimage(result_img_postproc).save(output_image_path)
		sess.close()

	print("Plot Losses and Gradients")
	Iterations = np.arange(args.max_iter)
	
	plt.ion()
	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
	ax1.plot(Iterations, style_loss_tab)
	ax1.set_ylabel("Style Loss")
	ax1.set_title('Comparison of the loss function')
	ax2.plot(Iterations, content_loss_tab)
	ax2.set_ylabel("Content Loss")
	ax3.plot(Iterations, loss_total_tab, color='r')
	ax3.set_ylabel("Total Loss")
	# Fine-tune figure; make subplots close to each other and hide x ticks for
	# all but bottom plot.
	f.subplots_adjust(hspace=0.2)
	#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
	f.savefig("LossFunction.png")
	#plt.figure
	f, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.plot(Iterations, grad_style_loss_tab)
	ax1.set_title('Comparison of the L2 norm of the gradients')
	ax1.set_ylabel('Grad Style Loss')
	ax2.plot(Iterations, grad_content_loss_tab)
	ax2.set_ylabel('Grad Content Loss')
	f.savefig("Gradient.png")
	
	f, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.plot(Iterations, grad_style_loss_min)
	ax1.plot(Iterations, grad_style_loss_max)
	ax1.set_title('Comparison of the range of the gradients')
	ax1.set_ylabel('Grad Style')
	ax2.plot(Iterations, grad_content_loss_min)
	ax2.plot(Iterations, grad_content_loss_max)
	ax2.set_ylabel('Grad Content')
	f.savefig("GradientMinMax.png")
	#TODO : regarder comment fonctionne LBFGS !! 
	print('np.mean(grad_style_loss_tab)',np.mean(grad_style_loss_tab),'np.mean(grad_content_loss_tab)',np.mean(grad_content_loss_tab))
	
	
	
	
	
		
def main():
	parser = get_parser_args()
	#style_img_name = "StarryNightBig"
	style_img_name = "StarryNight"
	content_img_name = "Louvre"
	max_iter = 1000
	print_iter = 100
	start_from_noise = 1 # True
	init_noise_ratio = 0.7
	content_strengh = 0.001
	# In order to set the parameter before run the script
	parser.set_defaults(style_img_name=style_img_name,max_iter=max_iter,
		print_iter=print_iter,start_from_noise=start_from_noise,
		content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,
		content_strengh=content_strengh)
	args = parser.parse_args()
	grad_computation(args)

if __name__ == '__main__':
	main()    
	
