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

In this scenario, the training policy will be trained with certain amount of images from dataset by dividing the total of the images by the batch size, the result of this division will be used to create batch training. By creating a loop from zero until reach batch training value, the sample of images will be taken with a certain range from the array with this calculation ```batch_training * BATCH_SIZE until (batch_training + 1) * BATCH_SIZE```. In the model context, the discriminator and generator will fine-tuned manually and the input of generater is ```512 x 4 x 4```

### Second Scenario
This scenario implemented in ```2. model.ipynb``` script. </br>

Same rule is applied in this scenario compared to the first scnario, however the numbers of trained data are controlled with ```tf.data.Dataset``` and due to some memory leakage the datatype is converted to lower floating point(from float32 to float16). In the model context, the discriminator and generator models compared to the first scenario will be bigger and using various advanced paramater such as epsilon and momentum in BN and kernel initializer to match the random noise distribution.

## Result and Analysis
### First Scenario
<b>Note : </b> The model configuration will be written inside with this format ```number_of_layer_stack x layer([parameter_inside:value])```.
<b>List of Abbreviation : </b> ```fs=filter size``` and ```ks=kernel size```
<details markdown="1">
<summary>Click here to see result summary</summary>
</br>
<table style="width:200%">
  <tr> 
      <th> Parameter </th>
      <th> Model Configuration </th>
      <th> Generator Loss </th>
      <th> Discriminator Real Loss </th>
      <th> Discriminator Fake Loss </th>
      <th> Result Sample </th>
  </tr>
  <tr> 
      <td> Adam learning rate:0.0003, beta:0.5, epoch:100, batch size:256</td>
      <td> Generator:</br> 4xTranspooseConv2D( </br>
        &ensp;[fs=32, ks=(4, 4)], </br>&ensp;[fs=64, ks=(4, 4)], </br>
        &ensp;[fs=128, ks=(4, 4)], </br>&ensp;[fs=3, ks=(4, 4)] </br>) </br>
        Discriminator:</br> 2xConv2D( </br>
        &ensp;[fs=32, ks=(3, 3)], </br>&ensp;[fs=32, ks=(3, 3)] </br>)</td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/195834317-13484e36-212a-4cda-b6eb-a12d417ce302.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/195834437-3e0cea19-dd3c-4e70-b734-2fef24fb1118.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/195834567-78dc1cff-d3e2-4ed7-8198-78c9eea0aeda.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/195834822-b3450d6e-f3fd-40cf-a8bd-677f90460ed3.jpg"/> </td>
  </tr>
  <tr> 
      <th> Parameter </th>
      <th> Model Configuration </th>
      <th> Generator Loss </th>
      <th> Discriminator Real Loss </th>
      <th> Discriminator Fake Loss </th>
      <th> Result Sample </th>
  </tr>
  <tr> 
      <td> Adam learning rate:0.0001, beta:0.5, epoch:100, batch size:256</td>
      <td> Generator:</br> 4xTranspooseConv2D( </br>
        &ensp;[fs=32, ks=(4, 4)], </br>&ensp;[fs=64, ks=(4, 4)], </br>
        &ensp;[fs=128, ks=(4, 4)], </br>&ensp;[fs=3, ks=(4, 4)] </br>) </br>
        Discriminator:</br> 2xConv2D( </br>
        &ensp;[fs=32, ks=(3, 3)], </br>&ensp;[fs=32, ks=(3, 3)] </br>)</td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196091721-d9f91e1b-98ee-43d3-afb1-b41d6b371375.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196091515-0ad40c0c-183b-4f68-9610-009cfc193fd3.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196091540-5640f56c-abb3-4d7b-bbcc-79f3cdac4d9d.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196091774-cdaa44aa-ce46-4f05-ba36-db0529ab5180.jpg"/> </td>
  </tr>
  <tr> 
      <th> Parameter </th>
      <th> Model Configuration </th>
      <th> Generator Loss </th>
      <th> Discriminator Real Loss </th>
      <th> Discriminator Fake Loss </th>
      <th> Result Sample </th>
  </tr>
  <tr> 
      <td> Adam learning rate:0.0002, beta:0.5, epoch:100, batch size:256</td>
      <td> Generator:</br> 4xTranspooseConv2D( </br>
        &ensp;[fs=32, ks=(4, 4)], </br>&ensp;[fs=64, ks=(4, 4)], </br>
        &ensp;[fs=128, ks=(4, 4)], </br>&ensp;[fs=3, ks=(4, 4)] </br>) </br>
        Discriminator:</br> 2xConv2D( </br>
        &ensp;[fs=32, ks=(3, 3)], </br>&ensp;[fs=32, ks=(3, 3)] </br>)</td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196096769-c6c599b3-21d6-4ddd-94f8-ecc8895fb152.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196096812-3deb84a1-8aa6-4c18-a289-35358abc9422.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196096843-97758c51-3777-473f-9e94-a2f327a93edb.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196096900-0c485720-394a-42e0-9006-03dd5a0d253c.jpg"/> </td>
  </tr>

### Second Scenario
