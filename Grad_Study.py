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
from decimal import Decimal

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
	weight_help_convergence = 1.
	content_loss = 0
	for layer, weight in content_layers:
		M = M_dict[layer[:5]]
		# If we add the M normalization we get a content loss smaller and so on a bigger emphasis on the style !
		P = tf.constant(dict_features_repr[layer])
		F = net[layer]
		content_loss +=  tf.nn.l2_loss(tf.subtract(P,F))* (weight*weight_help_convergence/(length_content_layers))
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
		grad = tf.subtract(P,F)* (2*weight/(length_content_layers))
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

def grad_stats_computation(sess,grad):
	#grad = tf.convert_to_tensor(grad)
	grad = grad[0]
	#grad = grad.eval(session=sess)
	l2_norm = tf.nn.l2_loss(grad)
	maximum = tf.reduce_max(tf.abs(grad))
	minimum = tf.reduce_min(tf.abs(grad))
	return(l2_norm,maximum,minimum)

def orthogonalite(A,B):
	""" A et B must be matrices """
	multiply = tf.matmul(tf.transpose(A),B)
	#print("multiply",multiply.shape)
	trace = tf.trace(multiply)
	#print("A.shape",A.shape)
	normeA = tf.norm(A,ord='fro',axis=[0,1])
	normeB = tf.norm(B,ord='fro',axis=[0,1])
	#print(trace)
	#print(normeA)
	cosAB = trace/(normeA*normeB)
	return(cosAB)
	
def ortogo_computation(sess,tensor_A,tensor_B):
	tabCosAB = np.zeros((3,1))
	tensor_A = tensor_A[0]
	tensor_B = tensor_B[0]
	tensor_A_eval = tensor_A
	#tensor_A_eval = tensor_A.eval(session=sess)
	#tensor_B_eval = tensor_B.eval(session=sess)
	tensor_B_eval = tensor_B
	#print("tensor_A.shape",tensor_A.shape)
	#A_slide = tensor_A_eval[0,:,:,0]
	#print("A_slide",A_slide.shape)
	for i in  range(3):
		# Different Channel
		A_slide = tensor_A_eval[0,:,:,i]
		B_slide = tensor_B_eval[0,:,:,i]
		#print("A_slide.shape",tensor_A_eval.shape)
		cosAB = orthogonalite(A_slide,B_slide)
		tabCosAB[i,:] = cosAB.eval(session=sess)
	return(tabCosAB)
		 
	
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
	weight_help_convergence = 1. # This wight come from a paper of Gatys
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

		# Propose different way to compute the lossses 
		weight_help_convergence_content = 10**5
		weight_help_convergence_style = 2*10**9
		ratio_weight_style_over_content = weight_help_convergence_style / ( weight_help_convergence_content * args.content_strengh)
		style_loss = sum_style_losses(sess,net,dict_gram,M_dict)
		content_loss =  sum_content_losses(sess, net, dict_features_repr,M_dict) # alpha/Beta ratio 
		loss_total = (args.content_strengh * weight_help_convergence_content) *content_loss  + weight_help_convergence_style * style_loss
		#grad_style_loss = grad_loss_style_norm(sess,net,dict_gram,M_dict)
		#grad_content_loss = grad_loss_content_norm(sess, net, dict_features_repr,M_dict)
		
		placeholder = tf.placeholder(tf.float32, shape=init_img.shape)
		assign_op = net['input'].assign(placeholder)
		length_tab = args.max_iter +1
		style_loss_tab = np.zeros((length_tab,1))
		content_loss_tab = np.zeros((length_tab,1))
		loss_total_tab =  np.zeros((length_tab,1))
		grad_style_loss_tab = np.zeros((length_tab,1))
		grad_content_loss_tab = np.zeros((length_tab,1))
		grad_style_loss_min = np.zeros((length_tab,1))
		grad_content_loss_min = np.zeros((length_tab,1))
		grad_style_loss_max = np.zeros((length_tab,1))
		grad_content_loss_max = np.zeros((length_tab,1))
		grad_total_loss_tab =  np.zeros((length_tab,1))
		grad_total_loss_max =  np.zeros((length_tab,1))
		grad_total_loss_min =  np.zeros((length_tab,1))
		grad_ortho =  np.zeros((length_tab,3))
		
		# Gradients
		variable = tf.trainable_variables() 
		grad_style_loss = tf.gradients(style_loss,variable)
		grad_content_loss = tf.gradients(content_loss,variable)
		grad_total_loss = tf.gradients(loss_total,variable)

		grad_stats_style = grad_stats_computation(sess,grad_style_loss)
		grad_stats_content = grad_stats_computation(sess,grad_content_loss)
		grad_stats_total = grad_stats_computation(sess,grad_total_loss)
		ortho_tb = []
		for j in range(3):
			ortho_tb +=  [orthogonalite(grad_content_loss[0][0,:,:,j],grad_style_loss[0][0,:,:,j])]

		
		if(args.optimizer=='GD'): # Gradient Descente		
			learning_rate = 10**(-10)
			optimizer = tf.train.GradientDescentOptimizer(learning_rate) # Gradient Descent
			train = optimizer.minimize(loss_total)
		elif(args.optimizer=='adam'):
			print('Adam')
			optimizer = tf.train.AdamOptimizer(args.learning_rate)
			grads_and_vars =  list(zip(grad_total_loss,variable))
			train = optimizer.apply_gradients(grads_and_vars)
	
		sess.run(tf.global_variables_initializer())
		sess.run(assign_op, {placeholder: init_img})
		
		sess.graph.finalize()
		print("sess.graph.finalize()")

		#sess.run(net['input'].assign(init_img)) # This line must be after variables initialization ! 
		t3 = time.time()
		print("sess initialized after ",t3-t2," s")
		print("loss before optimization")
		st.print_loss(sess,loss_total,content_loss,style_loss)
		for i in range(args.max_iter+1):
			style_loss_tab[i] = sess.run(style_loss)
			content_loss_tab[i] = sess.run(content_loss)
			loss_total_tab[i] = sess.run(loss_total)
			l2_norm,maximum,minimum = sess.run(grad_stats_style)
			grad_style_loss_min[i] = minimum
			grad_style_loss_max[i] = maximum
			grad_style_loss_tab[i] = l2_norm
			grad_content_loss_norml2,maximum,minimum = sess.run(grad_stats_content)
			grad_content_loss_tab[i] = grad_content_loss_norml2
			grad_content_loss_min[i] = minimum
			grad_content_loss_max[i] = maximum
			grad_total_loss_norml2,maximum,minimum = sess.run(grad_stats_total)
			grad_total_loss_tab[i] = grad_total_loss_norml2
			grad_total_loss_min[i] = minimum
			grad_total_loss_max[i] = maximum
			for j in range(3):
				grad_ortho[i,j] = sess.run(ortho_tb[j])
			
			if(i%args.print_iter==0):
				print("Iteration",i)
				st.print_loss(sess,loss_total,content_loss,style_loss)
				
			sess.run(train)
		
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
	Iterations = np.arange(length_tab)
	
	plt.ion()
	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
	ax1.semilogy(Iterations, style_loss_tab)
	ax1.set_ylabel("Style Loss")
	titre = 'Comparison of the loss function during {} Optimization, ratio = {:.2e}'.format(args.optimizer,Decimal(ratio_weight_style_over_content))
	ttl = ax1.set_title(titre)
	ttl.set_position([.5, 1.05])
	ax2.semilogy(Iterations, content_loss_tab)
	ax2.set_ylabel("Content Loss")
	ax3.semilogy(Iterations, weight_help_convergence_style *style_loss_tab, 'g^')
	ax3.semilogy(Iterations, args.content_strengh * weight_help_convergence_content * content_loss_tab, 'bs')
	ax3.semilogy(Iterations, loss_total_tab, color='r')
	ax3.legend(['Style','Content','Total'])
	ax3.set_ylabel("Total Loss")
	# Fine-tune figure; make subplots close to each other and hide x ticks for
	# all but bottom plot.
	f.subplots_adjust(hspace=0.2)
	#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
	f.savefig("LossFunction.png")
	#plt.figure
	f, (ax1, ax2,ax3) = plt.subplots(3, sharex=True)
	ax1.semilogy(Iterations, grad_style_loss_tab)
	titre = 'Comparison of the L2 norm of the gradients during {} Optimization, ratio = {:.2e}'.format(args.optimizer,Decimal(ratio_weight_style_over_content))
	ttl = ax1.set_title(titre)
	ttl.set_position([.5, 1.05])
	ax1.set_ylabel('Grad Style Loss')
	ax2.semilogy(Iterations, grad_content_loss_tab)
	ax2.set_ylabel('Grad Content Loss')
	ax3.semilogy(Iterations, grad_total_loss_tab, color='r')
	ax3.set_ylabel('Grad Total Loss')
	f.savefig("Gradient.png")
	f, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.plot(Iterations, grad_style_loss_min)
	ax1.plot(Iterations, grad_style_loss_max)
	titre = 'Comparison of the range of the gradients during {} Optimization'.format(args.optimizer)
	ttl = ax1.set_title(titre)
	ttl.set_position([.5, 1.05])
	ax1.set_ylabel('Grad Style')
	ax2.plot(Iterations, grad_content_loss_min)
	ax2.plot(Iterations, grad_content_loss_max)
	ax2.set_ylabel('Grad Content')
	f.savefig("GradientMinMax.png")
	#TODO :  add title method range formulat l2_loss + autre  
	print('np.mean(grad_style_loss_tab)',np.mean(grad_style_loss_tab),'np.mean(grad_content_loss_tab)',np.mean(grad_content_loss_tab))
	print("ratio_weight_style_over_content",ratio_weight_style_over_content)
	
	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
	ax1.plot(Iterations, grad_ortho[:,0])
	ax1.set_ylabel("B")
	titre = 'Cos of the Gradients during {} Optimization'.format(args.optimizer)
	ttl = ax1.set_title(titre)
	ttl.set_position([.5, 1.05])
	ax2.plot(Iterations, grad_ortho[:,1])
	ax2.set_ylabel("G")
	ax3.plot(Iterations, grad_ortho[:,2])
	ax3.set_ylabel("R")
	#print(grad_ortho)
	f.savefig("Cos_Gradient.png")
	input("Press Enter to end.")
	
	
	
	
		
def main():
	parser = get_parser_args()
	style_img_name = "StarryNight"
	content_img_name = "Louvre"
	max_iter = 1000
	print_iter = 100
	start_from_noise = 1 # True
	init_noise_ratio = 0.1
	content_strengh = 0.001
	optimizer = 'adam'
	# In order to set the parameter before run the script
	parser.set_defaults(style_img_name=style_img_name,max_iter=max_iter,
		print_iter=print_iter,start_from_noise=start_from_noise,
		content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,
		content_strengh=content_strengh,optimizer=optimizer)
	args = parser.parse_args()
	grad_computation(args)

if __name__ == '__main__':
	main()    
	
