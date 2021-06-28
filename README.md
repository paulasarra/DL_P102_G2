# DL_P103_G2
Emotion Based Face Generation GAN

dataset: https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset 

![gan_img](https://user-images.githubusercontent.com/48926447/122636581-56ac7700-d0ea-11eb-8d13-e248b4b59222.png)

## Motivation
When looking at all the possible topics, the face GAN generator was the one that we found the most interesting.  Moreover, we had heard about “this person does not exist”, a web page where every time you refresh a new person appears on the screen. We saw that it was done using the Generative Adversarial Network, and as we found it quite a challenge we wanted to try it.  

Also, the fact that the two models that GANs use are set up in a kind of “game” where the generator model seeks to fool the discriminator model, was a motivation for us since we wanted to know how does it work in practice. 

## The dataset
The dataset of this project contains 28811 black and white images and it is showing facial expressions. 
In order to classify the images into the categories we have used an emotions classifier which is an already trained model we found on github. Our first idea was to apply the classifier to the original dataset and then repeat the process after training the GAN,with the fake images.  But unfortunately this has not been possible and later on you will see why.

However, we have used the emotions classifier to group the images of the original dataset by the 7 different emotions, and obtained an accuracy of 90%. After the classification we have found an irregular amount of images in each of the categories but most of the emotions have more than 3000 images.

In this project we will be using the surprise sub dataset to compare the results that we will be obtaining in the different variations of the GAN, as we think is the one providing the best results.

## GAN
The objective is to generate new images that look like real faces. For that, we need a deep network in which we will input random noise and that is able to generate images from it. But how can we teach this network, which we call the generator,  to learn a meaningful mapping (so to go from something like noise to actual faces that seem real)? Here is when another network appears to help the generator, which is the discriminator network. How will we use it?
Well, we will input images into this network, but these images will come alternatively from two sources: one source will be the fake images coming from the generator and another source will be some images coming from a real dataset, so real faces. And the task of this discriminator is just to output if the image he receives is real or fake, so it is just like a binary classifier.
This combination is called an adversarial network because these two networks have opposite goals: the generator wants to fool the discriminator with his fake images, whereas the discriminator wants to realise when an image is fake.
In the end we have to think about the fact that we are in an unsupervised setting, we have  a lot of images but we have no labels, and having this discriminator network gives us a way to generate these labels. So that during the training the discriminator can tell the generator okay, now you are creating something close to a real face because it is getting more difficult for me to distinguish . And from that the generator will learn. 
So in practice, how will it be this process of learning? The discriminator will back propagate something, and this something will be a loss. 
And it is important to know that we have to be careful because if the discriminator gets too good from the beginning of the training, so it performs and distinguishes perfectly too early, then its loss will be zero and thus there will be nothing to backpropagate, no gradient, and as a consequence the generator would have no way to learn and improve. 

## Models and results
You can see the architecture of every different model in the slides 
### Model 1

As we know, in a traditional CNN we use pooling layers to downsample images. But for the generator, we need something like an inverse operation of this pooling layer. As Mireia explained, the generator network needs a layer to transform from a small input to a large and more detailed image output. 
So, a simple version of an unpooling is called an upsampling layer and it works by repeating the rows and columns of the input. Then, a more elaborate approach is to perform a backwards convolutional operation, which are these transposed convolutional layers. It is like a layer that combines the UpSampling2D and Conv2D layers into one layer: it both performs the upsample operation and interprets the input data to fill in the detail while it is upsampling.
And then we will also use the convolution to perform a form of downsampling by applying each filter across the input images. 
Moreover, in order to standardize layer outputs and thus stabilize the learning between the different layers, we will use batch normalization and the activation function after each layer will be a LeakyReLU. 
(Remember that we need these activation functions to avoid the fact that the entire network is equivalent to a single-layer model)
Then, to produce a fake image we will output from a sigmoid at the last layer. 
On the side of the discriminator, we will need to flatten our image before feeding it to the classifier. And finally, for the last dense layer we will use sigmoid in order to turn the values between 0 and 1 (fake or real).
This model has not given us good results. The faces are not even distinguishable.

### Model 2
This time we achieved better results than before. The shape of the faces can be appreciated, even though the eyes, nose,  and the other characteristics of a face are not well determined. 

Some changes were done from the previous architecture, now we didn’t use batch normalization nor upsampling.
The generator makes use of the Dense Layer as well as several Convolution transpose and the activation function LeakyReLU.

### Model 3
This model is exactly the same as before but using a much more simple architecture: there are 3 2d convolutional layers in the generator and 2 in the discriminator.

Moreover we have changed the RMS prop optimizer to the Adam optimizer. In this case we obtained better results not only in terms of computational efficiency but also regarding the quality of the images. Also we can start to see some blurry faces.

### Model 4
The fourth model is the first one we tried and it consists of the code provided by the list of topics. It was originally meant to generate new faces from a coloured dataset of celebrity faces. The first challenge we encountered was to adapt the code to our dataset of emotions, which is black and white, so basically changing the number of channels to 3.

The architecture of this model is similar to the previous ones, but in this case we do not use batch normalization or upsampling layers. Instead it alternates several convolutional layers with leakyReLu functions and the results obtained are the best ones we have seen so far. 
As you can see in the loss function, the generator loss starts with low values but keeps increasing while the discriminator is learning. We can clearly see a balance with the loss of the discriminator, that starts with high values and decreases till it reaches the value of 0.3-0.4.

### Model 5
Then, as we saw that with the last architecture we were starting to get good results, we wanted to keep it as a base, and work with it in order to improve our model. 
So, this new model will start with a Dense layer as well,  but with a smaller output size in order to include one transposed convolutional layer more in the model and still get the same size for the generated images. 
Then throughout the model the size of the outputs will be doubled by these transposed layers as Clara explained, while the channels will be the same size because we applied the padding  ‘same’. 
For the discriminator the structure will be more or less the same as before but we will include another convolutional layer as well, which will make the size of the flatten layer smaller. 
We see that the evolution of the loss is more or less the same that Paula explained, maybe it oscillates less, but the output images is a little bit cleaner than in the previous case.


### Model 6
And finally this model is the one in which we obtain the best results. We know they do not look like actual real faces but we see that the emotion of surprise is preserved and the performance compared with the first models has improved significantly. The difference between this model and the previous is that in this one we include another convolutional transposed layer more, both on the generator and the discriminator with all it involves regarding the sizes of the layers. 
As for the loss evolution, we see that now the discriminator has more trouble distinguishing between a real image and a fake one,  its loss is not going down as before. However, in both cases we can see the loss is oscillating a lot (which is something we will talk about in the conclusions) and that neither of them is outperforming the other, there is like a balance between the two, so that there is always new information for the generator to learn. 

### Conclusion
After running the different models we have noticed that we get quite unstable results in the generator. 

What could be happening is a mode collapse, which means that the generator model is only capable of generating a small subset of different outcomes, or modes. It is identified when the output images show low diversity between them or in our case when the loss plot presents oscillations over time, we have seen this most notably in the generator model.

ONE aspect that could be increasing the probability of having a mode collapse is the quality of our dataset. First of all, we are using black and white images of a small size, which can increase the difficulty of our training. Secondly, the different images in our dataset captured the faces from different perspectives. Also as we are training with the surprise images, some of the faces had a hand covering its mouth. And this could be a problem when determining the positioning of the objects. 

All in all, we have seen that GANs are powerful but complex and costly in time.





