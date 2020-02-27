"""
Created on 27/09/2019

This script have the goal to generate texture with different loss and 
parameters for the Journal paper 


@author: nicolas
"""

import Style_Transfer as st
from Arg_Parser import get_parser_args 
import os
import tensorflow as tf
import os.path
from shutil import copyfile
import pathlib
import time

moreSaveIm = '/home/gonthier/owncloud/These Gonthier Nicolas Partage/Images_Texturest_Résultats_More'


def get_list_of_images(path_origin):
	dirs = os.listdir(path_origin)
	dirs = sorted(dirs, key=str.lower)
	return(dirs)

def do_mkdir(path):
	if not(os.path.isdir(path)):
		os.mkdir(path)
	return(0)


	
def TEst():
	path_origin = '/media/gonthier/HDD2/Texture/ReferenceImage/'
	#path_origin = '/home/gonthier/Travail_Local/Texture_Style/Implementation Autre Algos/Subset'
	path_output = '/media/gonthier/HDD2/Texture/TexturesIM_output/AllResults/'
	path_output_tmp = '/media/gonthier/HDD2/Texture/TexturesIM_output/CompLoss/' # Sortie final
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 100
	print_iter = 100
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	beta_spectrum = 100
	alpha = 0.01
	list_img = get_list_of_images(path_origin)
	DrawAgain = False # Erase already synthesied image
	print(list_img)
	loss='autocorr'
	MSS =''
	padding = 'SAME'
	name_img = 'Tartan_2048.png'
	beta = 10**5
	saveMS=False
	for device in ["/cpu:0"]:
		t0 = time.time()
		with tf.device(device):
			print(device)
			MS_Strat = MSS
			name_img_wt_ext,_ = name_img.split('.')
			#path_output_tmp = path_output+name_img_wt_ext
			do_mkdir(path_output_tmp)
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding
			if not(beta==10**5):
				output_img_name += '_beta'+str(beta)
			for loss_item in loss:
				output_img_name += '_' + loss_item
			if 'spectrumTFabs' in loss:
				output_img_name += '_eps10m16'
			if not(MSS==''):
				output_img_name += '_MSS' +MSS
				if not(K==2):
					output_img_name += 'K' +str(K)
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding,MS_Strat=MS_Strat,
				saveMS=saveMS,beta=beta)
			args = parser.parse_args()
			output_img_name_full = path_output + output_img_name + '.png'
			st.style_transfer(args)
		t1 = time.time()
	print('Duration for',device,':',t1-t0,'s')  

def correctionName_betaFiles():
	"""
	The goal is just to correct the beta files images name
	"""
	path_base  = os.path.join(os.sep,'media','gonthier','HDD')
	owncloud_str ='owncloud'
	RefDir = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Reference')
	RefDir1024 = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Reference','1024')
	RefDir2048 = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Reference','2048')
	path_output_owncloud = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Output')
	path_output_save_aCopy1024 = os.path.join(path_output_owncloud,'1024')
	path_output_save_aCopy1024Beta = os.path.join(path_output_owncloud,'1024_Beta')
	path_output = '/media/gonthier/HDD2/Texture/TexturesIM_output/AllResults/'
	loss = ['Gatys','spectrumTFabs']
	scalesStrat = ['Init','']
	#MSS = 'Init'
	padding = 'SAME'
	K = 2
	list_img = get_list_of_images(RefDir1024)
	for MSS in scalesStrat:
		for name_img in list_img:
			if 'spectrumTFabs' in loss:
				beta_list = [0.1,1,10,100,1000,10000,10**5,100000000] # 10**8
				# beta_list = [10**5] # 10**8
			else:
				beta_list = [10**5]
			for beta in beta_list:
				print('##',name_img)
				name_img_wt_ext,_ = name_img.split('.')
				#path_output_tmp = os.path.join(path_output_owncloud1024,name_img_wt_ext)
				#path_output_tmp = path_output
				#path_output_tmp = os.path.join(path_output_allresults1024,name_img_wt_ext)
				#do_mkdir(path_output_tmp)
				#tf.reset_default_graph() # Necessity to use a new graph !! 
				#img_folder = RefDir1024
				#img_output_folder = path_output
				output_img_name = name_img_wt_ext + '_'+padding
				potential_img_name = name_img_wt_ext + '_'+padding
				output_img_name += '_beta'+str(beta)
				for loss_item in loss:
					output_img_name += '_' + loss_item
					potential_img_name += '_' + loss_item
				if 'spectrumTFabs' in loss:
					output_img_name += '_eps10m16'
					potential_img_name += '_eps10m16'
				if not(MSS==''):
					output_img_name += '_MSS' +MSS
					potential_img_name += '_MSS' +MSS
					if not(K==2):
						output_img_name += 'K' +str(K)
						potential_img_name += 'K' +str(K)
				potential_img_name += '_beta'+str(beta)
				
				# All resultats Image
				image_in_allres_Folder = os.path.join(path_output,name_img_wt_ext,potential_img_name + '.png')
				image_in_allres_Folder_copyIm = os.path.join(path_output,name_img_wt_ext,output_img_name + '.png')
				if os.path.exists(image_in_allres_Folder):
					copyfile(image_in_allres_Folder, image_in_allres_Folder_copyIm)  # src,dst
					if beta==10**5:
						image_in_allres_Folder_copyIm = image_in_allres_Folder_copyIm.replace('_beta'+str(beta),'')
						copyfile(image_in_allres_Folder, image_in_allres_Folder_copyIm) 
						
				# Folder for papers images
				image_in_owncloud = os.path.join(path_output_save_aCopy1024Beta,name_img_wt_ext,potential_img_name + '.png')
				image_in_owncloud_rename = os.path.join(path_output_save_aCopy1024Beta,name_img_wt_ext,output_img_name + '.png')
				print('Potential name : ',image_in_owncloud)
				print('new name :',image_in_owncloud_rename)
				if os.path.exists(image_in_owncloud):
					os.rename(image_in_owncloud, image_in_owncloud_rename)  # src,dst	
			
	
	
	
	
			
def generation_Texture():
	path_base  = os.path.join('C:\\','Users','gonthier')
	owncloud_str = 'Owncloud'
	onCluster = False
	if not(os.path.exists(path_base)):
		path_base  = os.path.join(os.sep,'media','gonthier','HDD')
		owncloud_str ='owncloud'
	if os.path.exists(path_base):
		RefDir = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Reference')
		RefDir1024 = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Reference','1024')
		RefDir2048 = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Reference','2048')
		path_output_allresults1024 = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','Images Textures Résultats')
		path_output_allresults2048 = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','HDImages_results')
		path_output_owncloud = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Output')
		path_output_save_aCopy1024 = os.path.join(path_output_owncloud,'1024')
		path_output_save_aCopy1024Beta = os.path.join(path_output_owncloud,'1024_Beta')
		path_output_save_aCopy2048 = os.path.join(path_base,owncloud_str,'2048')
		path_output_save_aCopy2048Beta = os.path.join(path_base,owncloud_str,'2048_Beta')
	else: # We certainly are on the cluster 
		RefDir = os.path.join(path_base,owncloud_str,'These Gonthier Nicolas Partage','ForTexturePaper','Reference')
		RefDir1024 = os.path.join('Images1024')
		RefDir2048 = os.path.join('HDImages2')
		path_output_allresults1024 = os.path.join('Images1024_output')
		path_output_allresults2048 = os.path.join('HDImages2_output')
		onCluster = True
		
	#path_origin = '/media/gonthier/HDD2/Texture/ReferenceImage/'
	#path_origin = '/home/gonthier/Travail_Local/Texture_Style/Implementation Autre Algos/Subset'
	path_origin = RefDir1024
	if not(onCluster):
		path_output = '/media/gonthier/HDD2/Texture/TexturesIM_output/AllResults/'
	else: # On cluster
		path_output = 'images_output'
	#path_output_tmp = '/media/gonthier/HDD2/Texture/TexturesIM_output/CompLoss/' # Sortie final
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 2000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	beta_spectrum = 100
	alpha = 0.01
	list_img = get_list_of_images(RefDir1024)
	DrawAgain = False # Erase already synthesied image
	print(list_img)
	
	# Plot with the different loss we proposed 
	# we want to save the intermediate image if we haven't run the image already
	saveMS = True
	losses_to_test = [['Gatys'],['Gatys','spectrumTFabs'],['autocorr']]
	scalesStrat = ['Init','']
	padding = 'SAME'
	K = 2
	for loss in losses_to_test:
		for MSS in scalesStrat:
			for name_img in list_img:
				if 'spectrumTFabs' in loss:
					beta_list = [0.1,1,10,100,1000,10000,10**5,100000000] # 10**8
					# beta_list = [10**5] # 10**8
				else:
					beta_list = [10**5]
				for beta in beta_list:
					MS_Strat = MSS
					print(name_img)
					name_img_wt_ext,_ = name_img.split('.')
					#path_output_tmp = os.path.join(path_output_owncloud1024,name_img_wt_ext)
					#path_output_tmp = path_output
					path_output_tmp = os.path.join(path_output_allresults1024,name_img_wt_ext)
					do_mkdir(path_output_tmp)
					tf.reset_default_graph() # Necessity to use a new graph !! 
					img_folder = RefDir1024
					img_output_folder = path_output
					output_img_name = name_img_wt_ext + '_'+padding
					if not(beta==10**5):
						output_img_name += '_beta'+str(beta)
					for loss_item in loss:
						output_img_name += '_' + loss_item
					if 'spectrumTFabs' in loss:
						output_img_name += '_eps10m16'
					if not(MSS==''):
						output_img_name += '_MSS' +MSS
						if not(K==2):
							output_img_name += 'K' +str(K)
					parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
						img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
						init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
						optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
						vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding,MS_Strat=MS_Strat,
						saveMS=saveMS,beta=beta)
					args = parser.parse_args()
					whereImageCreated = os.path.join(path_output,output_img_name + '.png')
					output_img_name_full =  os.path.join(path_output_allresults1024,name_img_wt_ext,output_img_name + '.png')
					# output folder / name texture / new texture . png
					if DrawAgain or not(os.path.isfile(output_img_name_full)):
						st.style_transfer(args)
						src=whereImageCreated
						dst1 = output_img_name_full
						copyfile(src, dst1)
						if not(onCluster):
							if 'spectrumTFabs' in loss:
								if beta == 10**5:
									dst2 = os.path.join(path_output_save_aCopy1024,name_img_wt_ext,output_img_name + '.png')
									copyfile(src, dst2)
									dst3 = os.path.join(path_output_save_aCopy1024Beta,name_img_wt_ext,output_img_name + '.png')
									destinationIm = dst3.replace(padding,padding+'_beta100000')
									#wholeName,ext = dst3.split('.')
									#wholeName += '_beta100000.png'
									copyfile(src, wholeName)
								else:
									dst2 = os.path.join(path_output_save_aCopy1024Beta,name_img_wt_ext,output_img_name + '.png')
									copyfile(src, dst2)
							else:
								dst2 = os.path.join(path_output_save_aCopy1024,name_img_wt_ext,output_img_name + '.png')
								copyfile(src, dst2)
					else:
						print(output_img_name_full,'already exists')

	# Run on the HD image in 2048*2048
	saveMS = True
	losses_to_test = [['Gatys'],['Gatys','spectrumTFabs']]
	scalesStrat = ['Init']
	padding = 'SAME'
	K = 3
	for loss in losses_to_test:
		for MSS in scalesStrat:
			for name_img in list_img:
				if 'spectrumTFabs' in loss:
					beta_list = [0.1,1,10,100,1000,10000,10**5,100000000] # 10**8
					# beta_list = [10**5] # 10**8
				else:
					beta_list = [10**5]
				for beta in beta_list:
					MS_Strat = MSS
					print(name_img)
					name_img_wt_ext,_ = name_img.split('.')
					#path_output_tmp = os.path.join(path_output_owncloud1024,name_img_wt_ext)
					#path_output_tmp = path_output
					path_output_tmp = os.path.join(path_output_allresults2048,name_img_wt_ext)
					do_mkdir(path_output_tmp)
					tf.reset_default_graph() # Necessity to use a new graph !! 
					img_folder = RefDir2048
					img_output_folder = path_output
					output_img_name = name_img_wt_ext + '_'+padding
					if not(beta==10**5):
						output_img_name += '_beta'+str(beta)
					for loss_item in loss:
						output_img_name += '_' + loss_item
					if 'spectrumTFabs' in loss:
						output_img_name += '_eps10m16'
					if not(MSS==''):
						output_img_name += '_MSS' +MSS
						if not(K==2):
							output_img_name += 'K' +str(K)
					parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
						img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
						init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
						optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
						vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding,MS_Strat=MS_Strat,
						saveMS=saveMS,beta=beta)
					args = parser.parse_args()
					whereImageCreated = os.path.join(path_output,output_img_name + '.png')
					output_img_name_full =  os.path.join(path_output_allresults2048,name_img_wt_ext,output_img_name + '.png')
					if DrawAgain or not(os.path.isfile(output_img_name_full)):
						st.style_transfer(args)
						src=whereImageCreated
						dst1 = output_img_name_full
						copyfile(src, dst1)
						if not(onCluster):
							if 'spectrumTFabs' in loss:
								if beta == 10**5:
									dst2 = os.path.join(path_output_save_aCopy2048,name_img_wt_ext,output_img_name + '.png')
									copyfile(src, dst2)
									dst3 = os.path.join(path_output_save_aCopy2048Beta,name_img_wt_ext,output_img_name + '.png')
									wholeName,ext = dst3.split('.')
									wholeName += '_beta100000.png'
									copyfile(src, wholeName)
								else:
									dst2 = os.path.join(path_output_save_aCopy2048Beta,name_img_wt_ext,output_img_name + '.png')
									copyfile(src, dst2)
							else:
								dst2 = os.path.join(path_output_save_aCopy2048,name_img_wt_ext,output_img_name + '.png')
								copyfile(src, dst2)
					else:
						print(output_img_name_full,'already exists')

def CopyAndTestImages():
	
	# cela marche pour les images en 1024*1024 pour celle en 2000 par 2000 il faudra faire differenmment car tu as fait un scale de plus
	path_base  = os.path.join('C:\\','Users','gonthier')
	
	listofmethod1024 =['_EfrosLeung','_EfrosFreeman','_SAME_Gatys','_SAME_Gatys_MSSInit','_SAME_Gatys_spectrumTFabs_eps10m16',\
			'_SAME_Gatys_spectrumTFabs_eps10m16_MSSInit','_DCor','_SAME_autocorr','_SAME_autocorr_MSSInit','MultiScale_o5_l3_8_psame',\
			'_TextureNets']
	listofmethod2048 =['_SAME_Gatys','_SAME_Gatys_MSSInitK3','_SAME_Gatys_spectrumTFabs_eps10m16_MSSInitK3',\
			'MultiScale_o6_l3_8_psame']
	
	for size in [1024,2048]:
		if not(os.path.exists(path_base)):
			path_base  = os.path.join(os.sep,'media','gonthier','HDD')
		if os.path.exists(path_base):
			RefDir = os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','ForTexturePaper','Reference',str(size))
			directory = os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','ForTexturePaper','Output',str(size))
			directory_forBeta = os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','ForTexturePaper','Output',str(size)+'_Beta')
		pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
		pathlib.Path(directory_forBeta).mkdir(parents=True, exist_ok=True)
		if size == 1024:
			listofmethod =listofmethod1024
		elif size==2048:
			listofmethod =listofmethod2048
		
		# List of the potential folder that can contain the images of interest
		potential_path_location = [os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','Images Textures Résultats'),\
			os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','Images Textures References Subset beta Variation'),\
			os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','Images_Texturest_Résultats_More'),\
			os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','HDImages_results'),\
			os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','HDImages_results_beta_variation'),\
			os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','Ulyanov_texturenets_said')]
		
		extension = ".png"
		ref_files = [file for file in os.listdir(RefDir) if file.lower().endswith(extension)]
		
		for file in ref_files:
			for method in listofmethod:
				if 'spectrumTFabs' in method:
					beta_list = ['_beta0.1','_beta1','_beta10','_beta100','_beta1000','_beta10000','','_beta100000000'] # 10**8
				else:
					beta_list = ['']
				for betastr in beta_list:
					filewithoutext = '.'.join(file.split('.')[:-1])
					stringname = filewithoutext + method + betastr
					stringnamepng = stringname + '.png'
					if 'spectrumTFabs' in method:
						if betastr=='':
							dst = os.path.join(directory,filewithoutext)
							pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
							dst = os.path.join(dst,stringnamepng)
							
							wholeName,ext = stringnamepng.split('.')
							wholeName += '_beta100000' +ext
							dst_beta = os.path.join(directory_forBeta,filewithoutext)
							pathlib.Path(dst_beta).mkdir(parents=True, exist_ok=True)
							dst_beta = os.path.join(dst_beta,wholeName)
						else:
							dst_beta = os.path.join(directory_forBeta,filewithoutext)
							pathlib.Path(dst_beta).mkdir(parents=True, exist_ok=True)
							dst_beta = os.path.join(dst_beta,stringnamepng)
							dst = None
					else:
						dst = os.path.join(directory,filewithoutext)
						pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
						dst = os.path.join(dst,stringnamepng)
						dst_beta = None
					if (not(dst is None)):
						dst = dst.replace('TextureNets','Ulyanov')
						dst = dst.replace('MultiScale','_Snelgorove_MultiScale')
					if (not(dst is None) and os.path.isfile(dst)) or (not(dst_beta is None) and os.path.isfile(dst_beta)):
						pass # The file already exist and it is in the right folder
					else:
						fileexist = False
						for path in potential_path_location:
							src = os.path.join(path,stringnamepng)
							# print(src)
							if os.path.isfile(src):
								if not(dst is None):
									copyfile(src,dst)
								if not(dst_beta is None):
									copyfile(src,dst_beta)
								fileexist = True
								pass
							src = os.path.join(path,filewithoutext,stringnamepng)
							# print(src)
							if os.path.isfile(src):
								if not(dst is None):
									copyfile(src,dst)
								if not(dst_beta is None):
									copyfile(src,dst_beta)
								fileexist = True
								pass
						if not(fileexist):
							print(stringnamepng,'doesn t exist !')
	
					

if __name__ == '__main__':
	# A faire : une fonction pour synthetiser les textures qui nous interesse (les differentes loss et les differentes valeurs de beta)
	# une fonction qui evalue les textures que nous n'avons pas encore faitees
	#CopyAndTestImages()
	generation_Texture()
	#TEst()
	#correctionName_betaFiles()
