import os
import glob

list_folder = glob.glob('*')

for folder in list_folder:
	print('folder :',folder)
	if not('old' in folder):
		list_im = glob.glob(folder+'/*.png')
		if '.png' in folder:
			list_im = [folder]
		for im in list_im:
			if '_spectrumGang' in im:
				tab = im.split('_')
				for i,elt in enumerate(tab):
					if elt=='spectrumGang':
						tab[i] = 'spectrumTFabs'
					if elt=='spectrumGang.png':
						tab[i] = 'spectrumTFabs.png'
				new_name = '_'.join(tab)
				print(im,'=>',new_name)
				os.rename(im,new_name)
				
for folder in list_folder:
	print('folder :',folder)
	if not('old' in folder):
		list_im = glob.glob(folder+'/*.png')
		if '.png' in folder:
			list_im = [folder]
		for im in list_im:
			if '_texture' in im:
				tab = im.split('_')
				for i,elt in enumerate(tab):
					if elt=='texture':
						tab[i] = 'Gatys'
					if elt=='texture.png':
						tab[i] = 'Gatys.png'
				new_name = '_'.join(tab)
				print(im,'=>',new_name)
				os.rename(im,new_name)

