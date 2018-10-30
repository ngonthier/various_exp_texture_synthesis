# Style-Transfer

This project implement with Tensorflow the Gatys Style Transfer algorithm.
This code can also be used for Texture generation in order to test new kind of loss function.
It is the projet for my master degree internship.

This code is inspired from :
- https://github.com/cysmith/neural-style-tf
- https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb


## Requirement 

If you want to have all the right requirements. We advice you to create a new enviromment (for instance conda one)
And then in this enviromment run :
pip install -r requirements.txt
If you don't have the admin right try :
pip install --user -r requirements.txt

For instance you need :
Tensorflow >= 1.2

You have to download the weights for the network :
- VGG19 CNN weights (It can be downloaded here : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)
- VGG19 normalized weights (can be download here : https://partage.mines-telecom.fr/index.php/s/DVw31LQ1UC0EoUM)


## Argument of the function

To synthesis a texture you have to run :
python Style_Transfer.py 

To get the help you can run : 
python Style_Transfer.py --help

## Test

I advice you to run the python script Test.py to try two synthesis 3 differents images
to test your installation.

