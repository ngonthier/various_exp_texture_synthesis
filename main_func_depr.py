#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  main_func_depr.py
#  
#  Copyright 2017 Nicolas <nicolas@Clio>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


def sum_style_losses(args,sess, net,img_art):
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
	total_style_loss = 0.	
	weight_help_convergence = 10**(9) # This wight come from a paper of Gatys
	sess.run(net['input'].assign(img_art))
	style_loss = 0.
	if(not(len(args.style_layers)==len(args.style_layer_weights))):
		if(args.verbose): print("Length of Style layer weight is not the same size as layer")
		weights = [1./len(args.style_layers)]*len(args.style_layers)
	else:
		weights = args.style_layer_weights / np.sum(args.style_layer_weights) # normalized
	for layer, weight in zip(args.style_layers, weights):
		a = sess.run(net[layer])
		x = net[layer]
		a = tf.convert_to_tensor(a)
		_, h, w, d = a.get_shape()
		M_a = h.value * w.value
		_, h, w, d = x.get_shape()
		M_x = h.value * w.value
		N = d.value
		A = gram_matrix(a, N,M_a)
		G = gram_matrix(x, N, M_x)
		style_loss = tf.nn.l2_loss(tf.subtract(G,A))  # output = sum(t ** 2) / 2
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2))
		total_style_loss += style_loss	
	return(total_style_loss)

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
