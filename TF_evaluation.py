#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  TF_evaluation.py
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

import numpy as np
import tensorflow as tf

def test_fft2d_of_tensorflow():
	#a = np.mgrid[:5, :5][0]
	#print(a)
	#np_fft_a = np.fft.fft2(a)
	#print(np_fft_a)
	#array([[ 50.0 +0.j        ,   0.0 +0.j        ,   0.0 +0.j        ,
          #0.0 +0.j        ,   0.0 +0.j        ],
       #[-12.5+17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,
          #0.0 +0.j        ,   0.0 +0.j        ],
       #[-12.5 +4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,
          #0.0 +0.j        ,   0.0 +0.j        ],
       #[-12.5 -4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,
            #0.0 +0.j        ,   0.0 +0.j        ],
       #[-12.5-17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,
          #0.0 +0.j        ,   0.0 +0.j        ]])
    # check if np.fft2d of TF.fft2d and NP have the same result

	testimage = np.random.rand(256, 256)
	testimage = testimage+0j

	ft_testimage = np.fft.fft2(testimage)
	np_result = np.sum(ft_testimage)
	print(np_result)

	tf_ft_testimage = tf.fft2d(testimage)
	tf_result = np.sum(tf_ft_testimage.eval())
	print(tf_result)

	result_div = np.abs(tf_ft_testimage.eval())

	plt.imshow(np.log(result_div))

	print(np_result)
	(56368.5840888+9.09494701773e-13j)

	print(tf_result)
	(56368.6+0.00390625j)
    
     

def main(args):
    return 0

if __name__ == '__main__':
    test_fft2d_of_tensorflow()
