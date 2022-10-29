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
</table>
</details>
&emsp; The test is conducted with different learning rate simultaneously 0.0003, 0.0001 and 0.0002 and using Adam optimizer with beta parameter of 0.5. With the parameter of 0.0003 the fluctuation of each loss is very noisy, inconsistent and the trend is increasing especially in generator loss which resulting in rough color gradient images. While for the 0.0001 the overall trend is very stable did not improve yet the noises is significantly decreased compared to learning rate of 0.0003, Additionally resulting in way softer color gradient. The last one which is 0.0002 the trend of generator is increases over time and not very noisy, while the discriminator in late epoch started to decreases over a time, overall the fluctuation is steady and did not fluctuative, The generated result, not only the color gradient is softer the model learn enough to form an image which way more close to a face than the previous learning rate. </br>
&emsp; In a summary, the learning rate of 0.0002 is the most stable learning rate in case of GAN training, the model did not stuck to learn compared to 0.0001 and the color gradient way more softer than 0.0003. However further consideration to balance the model is required in order to create a strong adversariality which will improve the result. In addition, the training function may also need to be updated especially for sample selection based on the batch size.

### Second Scenario
<b>Note : </b> in this scenario, the only parameter that been changed are ```batch size, addition of batch norm(momentum and epsilon) and weight initialization```.</br>
<b>Additional Note : </b> The average of good sample is calculated by : ```sum of [defined_type] sample / total generated sample``` </br>
<b> Defined Sample Type : </b> </br>
Green = Good Quality (GQ) <b> where : </b> overall of the image is recognizable as face and less distorted</br>
Blue = Decent Quality (DQ) <b> where : </b> overall of the image is recognizable as face and slight distorted</br>
Red = Poor Quality (PQ) <b> where : </b> overall of the image is not recognizable as face and sometimes very distorted</br>
<details markdown="1">
<summary>Click here to see result summary</summary>
</br>
<table style="width:200%">
  <tr> 
      <th> Figure Number </th>
      <th> Parameter Configuration </th>
      <th> Generator and Discriminator Loss </th>
      <th> Result Sample </th>
      <th> Average Good Sample </th>
  </tr>
  <tr>
      <td>Figure 1 </td>
      <td>
        <b>General Param </b>: batch size:10, epoch:40 </br>
        <b>Generator     </b>: truncated normal(stddev) weight init:0.02, generator batch norm(momentum; epsilon):0.8; 10**5 </br> 
        <b>Discriminator </b>: discriminator batch norm(momentum):0.8 </br>
      </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196102991-2be0bffe-c999-465b-8015-2a4a1961c29f.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196392402-9ce60b18-da3c-4482-b203-f2af0a2bddb9.jpg"/> </td>
      <td> 
        GQ Rate: </br>3 / 20  = <b>15%</b> </br>
        DQ Rate: </br>7 / 20  = <b>35%</b> </br>
        PQ Rate: </br>10 / 20 = <b>50%</b>
      </td>

  </tr>
  <tr>
      <td>Figure 2 </td>
      <td>
        <b>General Param </b>: batch size:64, epoch:40 </br>
        <b>Generator     </b>: truncated normal(stddev) weight init:0.02, generator batch norm(momentum; epsilon):0.8; 10**5 </br> 
        <b>Discriminator </b>: discriminator batch norm(momentum):0.8 </br>
      </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196103653-0acb5c9f-d21c-4932-a551-16047f797b67.png"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196393687-e97714d4-2d35-45ed-b649-26384365c805.jpg"/> </td>
      <td> 
        GQ Rate: </br>3 / 20  = <b>15%</b> </br>
        DQ Rate: </br>2 / 20  = <b>10%</b> </br>
        PQ Rate: </br>15 / 20 = <b>75%</b>
      </td>
  </tr>
  <tr> 
      <td>Figure 3 </td>
      <td>
        <b>General Param </b>: batch size:64, epoch:40 </br>
        <b>Generator     </b>: truncated normal(stddev) weight init:0.02 </br> 
        <b>Discriminator </b>: discriminator batch norm(momentum):0.8 </br>
      </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196108877-23628a17-5480-4fc7-b3be-6aadb2f678fb.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/196393826-f8ab1a4e-98f1-4620-9edf-10cd851ade75.jpg"/> </td>
      <td> 
        GQ Rate: </br>1 / 20  = <b>5%</b> </br>
        DQ Rate: </br>4 / 20  = <b>40%</b> </br>
        PQ Rate: </br>15 / 20 = <b>75%</b>
      </td>
  </tr>
  <tr>
      <td>Figure 4 </td>
      <td>
        <b>General Param </b>: batch size:64, epoch:200 </br>
        <b>Generator     </b>: truncated normal(stddev) weight init:0.02, increasing conv filter to (128, 64, 64) </br> 
        <b>Discriminator </b>: discriminator batch norm(momentum):0.8, decreasement conv filter to (32, 64, 64, 128) </br>
      </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/198332601-d875484d-8c72-42ef-a5b8-9451cacea7a2.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/198331231-7c1cd038-eacf-46a3-a05f-0327fce1788f.jpg"/> </td>
      <td> 
        GQ Rate: </br>3 / 20  = <b>15%</b> </br>
        DQ Rate: </br>8 / 20  = <b>40%</b> </br>
        PQ Rate: </br>9 / 20 = <b>45%</b>
      </td>
  </tr>
  <tr>
      <td>Figure 5 </td>
      <td>
        <b>General Param </b>: batch size:64, epoch:200 </br>
        <b>Generator     </b>: truncated normal(stddev) weight init:0.02, increasing conv filter to (128, 128, 64) </br> 
        <b>Discriminator </b>: discriminator batch norm(momentum):0.8, decreasement conv filter to (32, 64, 64, 128) </br>
      </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/198333812-38339ea5-b63c-4e0d-988e-06bd342924f4.jpg"/> </td>
      <td> <img src="https://user-images.githubusercontent.com/54882818/198333889-527964ec-b65d-4796-b7f5-6791710e36ef.jpg"/> </td>
      <td> 
        GQ Rate: </br>5 / 20  = <b>25%</b> </br>
        DQ Rate: </br>9 / 20  = <b>45%</b> </br>
        PQ Rate: </br>6 / 20 = <b>30%</b>
      </td>
  </tr>
</table>
</details>
&emsp; The result of the experiment in this section shows that, based on the generator loss and discriminator loss, in order to reach the nash equilibrium generator and discriminator must be in opposite way, generator must be reach local maxima and discriminator must reach local minima. However, in context of real application reaching nash equilibrium is very hard. Also, in the final result the overall good result shows when the discriminator and generator loss become really fluctuative and showing great adversariality as shown in figure 1, 4 and 5. Especially in Figure 4 and 5, the obvious adversariality has been shown there, the parameter tuning to balance discriminator and generator either by increasing or decreasing filter size affecting the result overall by most increasement to 25% good rate compared to the other. Batch size also affecting the result shows in figure 1 compared to 2 and 3 where lower batch size decreasing the variety of an anime faces which also increase the good result.

## Conclusion
&emsp; The result illustrates from first and second experiment in average Generative Adversarial Network still had a major difficulties to be consistently generate excellent result. Further parameter exploration and tuning might taking long time to enhance and in order to reach the state of good adversariality then reach nash equilibrium is still challenging and require further method or optimization to increase the quality of generated images. Mentionable parameter here that affecting the quality are batch size, number of convolutional and kernel initialization that match with generated latent spaces to decrease the variety and improve the quality of generated anime faces.
