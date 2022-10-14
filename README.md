# Anime Face picture generator using Generative Adversarial Network

## Abstract
&emsp; Generative Adversarial Network(GAN) is one of the basis method for deep fake by utilizing two different deep learning model and create an adversariality between these model. The first model called discriminator which have a role to discriminate/differentiate an image whether fake or real, the second model called as generator which the role is to generate the image using the given random noise. In a simple analogy, the generator have a role as a fake money maker while the discriminator is the police, both working together as an enemy. The discriminator will always tell the generator that the money he is making is still a fake and the generator will improve. However, both of the models has to be near as equally strong inorder to keep improving and generate similiar result from the dataset. In this anime cases, because the context is generating an images, the network will utilize convolutional neural network. The experiment will be conducted with two different training scenario in context of model and training rules and the result will be used as an analysis and conclude the characteristic of GAN. In addition, this repo in the future will improve by using different GAN scenario such as StyleGAN to generate better result.

## Dataset
There is two different anime face dataset that can be downloaded from kaggle  
1. www.kaggle.com/splcher/animefacedataset
2. www.kaggle.com/datasets/soumikrakshit/anime-faces

## Installation
- The project runs on Python 3.8 and developed with Jupyter notebook(.ipynb) extension.
- In order to run this project install this requirements
```
pip install -r requirements.txt
```

## Scenario
### First Scenario
This scenario implemented in ```1. model.ipynb``` script. </br>

In this scenario, the training policy will be trained with certain amount of images from dataset by dividing the total of the images by the batch size, the result of this division will be used to create batch training. By creating a loop from zero until reach batch training value, the sample of images will be taken with a certain range from the array with this calculation ```batch_training * BATCH_SIZE until (batch_training + 1) * BATCH_SIZE ```. In the model context, the discriminator and generator will fine-tuned manually and the input is of noise ```512 x 4 x 4```
### Second Scenario
This scenario implemented in ```2. model.ipynb``` script. </br>

Same rule is applied in this scenario compared to the first scnario, however the numbers of trained data are controlled with ```tf.data.Dataset``` and due to some memory leakage the datatype is converted to lower floating point(from float32 to float16). In the model context, the discriminator and generator models compared to the first scenario will be bigger and using various advanced paramater such as epsilon and momentum in BN and kernel initializer to match the random noise distribution.

## Result and Analysis

