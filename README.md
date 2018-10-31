
# Parallel Convolutional Neural Networks with Cuda
## Xiangkai Zeng (xiangkaz), Matthew Lipari (mlipari)

## URL
https://mattslipari.github.io/CudaCNN/

## SUMMARY

We plan to implement a parallel version of a Convolutional Neural Network on Nvidia GPUs using CUDA. This involves the forward computation and backpropagation of the convolutional layer, the fully connected layer and the pooling layer. We are going to start from scratch and write a simple sequential version first and then optimize it by parallel computation. We also plan to validate our implementation by analyzing its results on specific tasks.

## BACKGROUND

Deep Learning and Neural Networks are becoming increasingly popular in the areas of Machine Learning, Artificial Intelligence, Computer Vision, Natural Language Processing and many more. However, in order to train a deep neural network, you often need a large amount of data and time. Since many implementations for neural networks involve large matrix multiplications, people often use GPUs to train their models. 

A widely used neural network structure for Computer Vision and Natural Language Processing is called a Convolutional Neural Network. The major parts of it are the convolutional layer, pooling layer and fully connected layer. Similar to other layers in neural networks, training these parts are often time-consuming. Fortunately, different elements in the weight matrices can be computed in parallel. Therefore, GPUs are a good way to speedup the training and testing process of neural network models. 

In this project, we plan to implement parallel Convolutional Neural Network and run our implementation for a real life application. 

## THE CHALLENGE


Since we will be running our CNN on huge datasets, there are a number of challenges we expect to face:

We need to find a way to efficiently parallelize matrix multiplication. This is one of the most important and frequently used computations for the CNN.
For the backpropagation algorithm, it could happen that multiple threads try to update the same weight at a given time. Therefore, we need to ensure the correctness of this algorithm but also try to maintain the speedup of parallel computation.
Since the amount of data is large, we need to identify the temporal and spatial locality to better optimize our implementation.
Because images have channels besides rows and cols, it will actually be a three dimensional matrix. If we plan to use batch gradient descent algorithm, there is one more dimension for the batch size, which makes it a four dimension tensor. This will make the code more complicated and harder to optimize.

## RESOURCES
We will need computers with GPUs to run our experiment. For this we will use the same hardware that we used in Assignment 2 (GHC Machines with NVIDIA GeForce GTX 1080 GPUs). We will rely on the papers mentioned in the reference to help us optimize and complete our implementation.

## GOALS AND DELIVERABLES

### Plan to Achieve
We will create a parallel CNN that can handle basic image processing tasks. Our CNN will show marked improvement from a sequential implementation. Our implementation will be optimized from a variety of methods and we will assure the correctness of it by testing our model on a practical dataset.

### Hope to Achieve
We hope to be able to use our CNN for a very interesting image processing application. We have looked into both Image Style Transfer and Image Colorization. 
Image Style Transfer involves transferring the ‘style’ of one image onto another image. For example, using the iconic brush strokes of a van Gogh painting to make an ordinary image of an apple look as if it were painted by van Gogh himself.
Image Colorization involves the coloring of a black and white image to make it look as if it were originally captured as a colored image. Say we have some photo taken in the 19th century- what does the color version of this photograph look like? 
Of course, we hope to be able to train our CNN to handle basic cases for each of these image processing techniques, but in the end we would be satisfied if we were able to implement just one. There is a bit of research being done, or already done, on each of these applications, so we would likely focus on the one with more of an existing foothold.

## PLATFORM CHOICE
In order to efficiently parallelize our CNN, we need to have access to a very large number of threads. As a result, we have decided to utilize CUDA C++ because of the sheer number of threads it gives us access to.

## REFERENCES
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.

LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "Image style transfer using convolutional neural networks." 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2016.

Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European Conference on Computer Vision. Springer, Cham, 2016.

## SCHEDULE

**Week of** | **Goal** |
---| ---|
**November 3rd**  | Finish proposal, start writing Cuda code  | 
**November 10th** | Continue writing Cuda code  | 
**November 17th** | Finish initial Cuda implementation, test code |
**November 24th** | Optimize based on test performance |
**December 1st** | Attempt hopeful implementation |
**December 8th**  | Complete final version of code | 
**December 15th** | Finish poster and final report |
