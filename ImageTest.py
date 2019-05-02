#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ImageTest.py
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

import scipy.io
from PIL import Image
import Style_Transfer as st

def main(args):
    
    img_folder='images/'
    image_style_name= "StarryNight_Big"
    image_style_name= "StarryNight"
    starry = "StarryNight"
    marbre = 'GrungeMarbled0021_S'
    tile =  "TilesOrnate0158_1_S"
    tile2 = "TilesZellige0099_1_S"
    peddle = "pebbles"
    brick = "BrickSmallBrown0293_1_S"

    inputname = img_folder + brick + '.png'
    outputname = img_folder + brick + '_2' + '.png'
    
    img = scipy.misc.imread(inputname)
    img_prepross = st.preprocess(img.astype('float32'))
    #img = scipy.misc.imread(inputname)
    
    img_post = st.postprocess(img_prepross)
    
    scipy.misc.toimage(img_post).save(outputname)


    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
