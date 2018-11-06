# Style-Transfer

This project implement with Tensorflow the Gatys Style Transfer algorithm.
This code can also be used for Texture generation in order to test new kind of loss function.
It is the projet for my master degree internship.

This code is inspired from :
- https://github.com/cysmith/neural-style-tf
- https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb


## Requirement 

This code have been test on Python 3.5, it is probable that it doesn't work on Python 2.

If you want to have all the right requirements. We advice you to create a new enviromment (for instance conda one)
And then in this enviromment run :
pip install -r requirements.txt
If you don't have the admin right try :
pip install --user -r requirements.txt

This requirements file have been generated with the help of pipreqs.

For instance you need :
tensorflow-gpu >= 1.2

If you don't have a GPU supported by Tensorflow, replace the tensorflow-gpu>=1.2.0 line by tensorflow>=1.2.0

You have to download the weights for the network :
- VGG19 normalized weights (can be download here : https://partage.mines-telecom.fr/index.php/s/sqa9QntDcPqgLex)
Those weights are used by default in the code.

If you want to use the usual pretrained weights : 
- VGG19 CNN weights (It can be downloaded here : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)



## Argument of the function

To synthesis a texture you have to run :
python Style_Transfer.py 

To get the help you can run : 
python Style_Transfer.py --help

### Simple version of the command line

To synthesis a texture with the Gatys loss you have to run :
python Style_Transfer.py --verbose --style_img_name TilesOrnate0158_1_S --max_iter 1000 --loss Gatys --output_img_name Gatys_syn

For the Gatys plus spectrum :
python Style_Transfer.py --verbose --style_img_name TilesOrnate0158_1_S --max_iter 1000 --loss Gatys,spectrumTFabs --output_img_name Gatys_spectrum_syn

For autocorr :
python Style_Transfer.py --verbose --style_img_name TilesOrnate0158_1_S --max_iter 1000 --loss autocorr --output_img_name Autocorr_syn

For using the multi scale initialization strategy :
python Style_Transfer.py --verbose --style_img_name TilesOrnate0158_1_S --max_iter 1000 --loss Gatys --output_img_name Gatys_MSInit_syn --MS_Strat Init --K 2
   

## Test

I advice you to run the python script Test.py to try two synthesis 3 differents images
to test your installation.

