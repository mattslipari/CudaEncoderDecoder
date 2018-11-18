# Parallel Encoder Decoder Network with Cuda
Xiangkai Zeng (xiangkaz), Matthew Lipari (mlipari)

## URL
https://mattslipari.github.io/CudaEncoderDecoder/

## SUMMARY

We plan to implement a parallel version of an Encoder-Decoder Network on Nvidia GPUs using CUDA. This involves the forward computation and backpropagation of the Recurrent Neural Network (RNN), the fully connected layer and various different activation layers. The backpropagation of RNN is actually backpropagation through Time (BPTT), which makes it more complicated and harder to parallelize. We are going to start from scratch and write a simple sequential version first and then optimize it by parallel computation. We also plan to validate our implementation by analyzing its results on specific tasks.

## BACKGROUND

Deep Learning and Neural Networks are becoming increasingly popular in the areas of Machine Learning, Artificial Intelligence, Computer Vision, Natural Language Processing and many more. However, in order to train a deep neural network, you often need a large amount of data and time. Since many implementations for neural networks involve large matrix multiplications, people often use GPUs to train their models. 

A widely used neural network structure for Natural Language Processing and Computer Vision is called a Recurrent Neural Network. The original RNN suffers from the gradient vanishing and the gradient exploding problem. Both the  LSTM RNN and GRU RNN have been invented to overcome the problem. Similar to other layers in neural networks, training these parts is often time-consuming. Fortunately, different elements in the weight matrices can be computed in parallel. Therefore, GPUs are a good way to speedup the training and testing process of neural network models. Specifically, a special type of RNN is called Encoder-Decoder Network which is designed to address the sequence-to-sequence problems. It consists of two RNN and the outputs of the first RNN are the inputs to the second RNN, which makes it more complicated to implement and parallelize.

In this project, we plan to implement a parallel Encoder-Decoder Network and run our implementation for a real life application. 

## THE CHALLENGE

Since we will be running our RNN on huge datasets, there are a number of challenges we expect to face:

We need to find a way to efficiently parallelize matrix multiplication. This is one of the most important and frequently used computations for the RNN.
Since RNN have sequential dependency, we cannot parallel among different timesteps. Also, the Backpropagation through Time makes it more complicated to compute.
For the backpropagation algorithm, it could happen that multiple threads try to update the same weight at a given time. Therefore, we need to ensure the correctness of this algorithm but also try to maintain the speedup of parallel computation.
Since the amount of data is large, we need to identify the temporal and spatial locality to better optimize our implementation.
Because sequences such as sentences have words and words can be represented as vectors, it will actually be a two dimensional matrix. If we plan to use batch gradient descent algorithm, there is one more dimension for the batch size, which makes it a three dimension tensor. This will make the code more complicated and harder to optimize.

## RESOURCES
We will need computers with GPUs to run our experiment. For this we will use the same hardware that we used in Assignment 2 (GHC Machines with NVIDIA GeForce GTX 1080 GPUs). We will rely on the papers mentioned in the reference to help us optimize and complete our implementation.

## GOALS AND DELIVERABLES

### Plan to Achieve
We will create a parallel Encoder-Decoder RNN that can handle basic machine translation or similar sequential processing tasks. Our RNN will show marked improvement from a sequential implementation. Our implementation will be optimized from a variety of methods and we will assure the correctness of it by testing our model on a practical dataset.

### Hope to Achieve
We frankly do not know the difficulty of some of the things we plan to accomplish. One of the things we believe will be hard is adding in the ‘attention’ step to our Encoder-Decoder RNN. The attention step allows the model to give attention to specific inputs that it thinks are relevant for a given output. The attention step would likely increase the quality of our model, but whether or not it is feasible to get much parallelization is unknown to us at this time (we will update this proposal once we learn more information)

Another challenge is to implement multi-layer RNN on top of the basic encoder-decoder network. There is potential parallelization among layers, but it would also makes the model more difficult to implement. 

## PLATFORM CHOICE
In order to efficiently parallelize our Encoder-Decoder RNN, we need to have access to a very large number of threads. As a result, we have decided to utilize CUDA C++ because of the sheer number of threads it gives us access to.

## REFERENCES
Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to forget: Continual prediction with LSTM." (1999): 850-855.

Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).

Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks//Advances in neural information processing systems. 2014: 3104-3112.


## SCHEDULE

**By** | **Goal** |
---| ---|
**November 10th** | Finish proposal, start writing Cuda code for a basic RNN  | 
**November 17th** | Finish initial Cuda implementation, test code |
**November 19th** | Project Milestone |
**November 24th** | Use this code to start writing the Encoder-Decoder RNN |
**December 1st** | Test and optimize our code |
**December 8th**  | Complete final version of code | 
**December 15th** | Finish poster and final report |
