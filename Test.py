#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 2017

The goal of this script is to test some function from the project

@author: nicolas
"""

import Style_Transfer as st
import pickle
from Arg_Parser import get_parser_args 
import seaborn as sns
import scipy
import scipy.stats
import tensorflow as tf

def test_moments_computation():
	shape_tensor = (1,300,400,64)
	dist_names = ['expon', 'norm']
	sess = tf.Session()
	epsilon = 10**(-1)
	epsilonmean = 10**(-1)
	for dist_name in dist_names:
		print(dist_name)
		dist = getattr(scipy.stats, dist_name)
		tensor4D = dist.rvs(size=shape_tensor)
		mean_x,variance_x,skewness_x,kurtosis_x = st.compute_4_moments(tensor4D)
		assert sess.run(tf.shape(mean_x))[0] == shape_tensor[3]
		assert sess.run(tf.shape(variance_x))[0] == shape_tensor[3]
		assert sess.run(tf.shape(skewness_x))[0] == shape_tensor[3]
		assert sess.run(tf.shape(kurtosis_x))[0] == shape_tensor[3]
		mean, var, skew, kurt = dist.stats(moments='mvsk')	
		kurt += 3 # because fisher definition of the formula 
		mean_x = sess.run(mean_x)
		variance_x = sess.run(variance_x)
		skewness_x= sess.run(skewness_x)
		kurtosis_x = sess.run(kurtosis_x)
		testOk = True
		for i in range(shape_tensor[3]):
			if(abs(mean-mean_x[i])>epsilonmean):
				print("Mean Error",i,"mean theoritical = ",mean," mean computed = ",mean_x[i])
				testOk = False
			if(abs(var-variance_x[i])/var>epsilon):
				print("Var Error",i,"Var theoritical = ",var," var computed = ",variance_x[i])
				testOk = False
			if(abs(skew-skewness_x[i])>epsilonmean):
				print("skewness Error",i,"skewness theoritical = ",skew," skewness computed = ",skewness_x[i])
				testOk = False
			if(abs(kurt-kurtosis_x[i])/kurt>epsilon):
				print("kurtosis Error",i,"kurtosis theoritical = ",kurt," kurtosis computed = ",kurtosis_x[i])
				testOk = False
	if(testOk) : 
		print("Test OK")
	else:
		print("Test not OK")		
	sess.close()
		
def test_n_moments_computation():
	
	shape_tensor = (1,300,400,64)
	dist_names = ['expon', 'norm']
	sess = tf.Session()
	epsilon = 10**(-1)
	epsilonmean = 10**(-1)
	for dist_name in dist_names:
		print(dist_name)
		dist = getattr(scipy.stats, dist_name)
		tensor4D = dist.rvs(size=shape_tensor)
		mean_x,variance_x,skewness_x,kurtosis_x = st.compute_n_moments(tensor4D,4)
		assert sess.run(tf.shape(mean_x))[0] == shape_tensor[3]
		assert sess.run(tf.shape(variance_x))[0] == shape_tensor[3]
		assert sess.run(tf.shape(skewness_x))[0] == shape_tensor[3]
		assert sess.run(tf.shape(kurtosis_x))[0] == shape_tensor[3]
		mean, var, skew, kurt = dist.stats(moments='mvsk')	
		kurt += 3 # because fisher definition of the formula 
		mean_x = sess.run(mean_x)
		variance_x = sess.run(variance_x)
		skewness_x= sess.run(skewness_x)
		kurtosis_x = sess.run(kurtosis_x)
		testOk = True
		for i in range(shape_tensor[3]):
			if(abs(mean-mean_x[i])>epsilonmean):
				print("Mean Error",i,"mean theoritical = ",mean," mean computed = ",mean_x[i])
				testOk = False
			if(abs(var-variance_x[i])/var>epsilon):
				print("Var Error",i,"Var theoritical = ",var," var computed = ",variance_x[i])
				testOk = False
			if(abs(skew-skewness_x[i])>epsilonmean):
				print("skewness Error",i,"skewness theoritical = ",skew," skewness computed = ",skewness_x[i])
				testOk = False
			if(abs(kurt-kurtosis_x[i])/kurt>epsilon):
				print("kurtosis Error",i,"kurtosis theoritical = ",kurt," kurtosis computed = ",kurtosis_x[i])
				testOk = False
	if(testOk) : 
		print("Test OK pour n moments")
	else:
		print("Test not OK pour n moments")		
	sess.close()

if __name__ == '__main__':
	test_moments_computation()
	test_n_moments_computation()
