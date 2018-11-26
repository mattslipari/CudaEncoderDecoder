## Completed So Far
	
We have tested two different implementations for the Feed Forward Network, which is a basic component of our encoder-decoder network. Our first implementation is to use naive cuda code to do matrix and vector calculations. The second implementation is to use cuBLAS to speed up the computation. We found that cuBLAS was faster than our naive implementation, so we have decided to use cuBLAS in our project. Also, we noticed that we need to design different classes to make our code more clean and readable. We have had a lot of initial difficulty dealing with matrices of varying sizes and making sure that all of the dimensions are correct, so we know that making these classes will help us a lot.

## Goals
	
Although we are behind where we had initially indicated, we still believe that we will meet all of our “Plan to achieve” goals. We initially underestimated the amount of work we would need to do to build a strong foundation on which our RNN could be created and we also overestimated the amount of time we had available to work on these goals. Now we have more experience using our resources and understanding exactly what has to be done. We’ve also set up a schedule so that we can meet 3-4 times a week, so we should have no problem reaching the goals we initially set.  

We also believe that we will be able to do significant work towards our “Hope to achieve” goals. We set these goals to be challenging but not impossible- so we do not think that we need to change them yet. If we feel as if we need to change these goals, we will update the proposal accordingly.

## Poster Session

For the poster session, we plan on showing a demo of our code and results from our various implementation attempts. We believe that our neural network will be able to perform a simple, yet interesting, classification task, so that we can show that our code actually does something useful and is not just a jumble of computation. In addition to this, we would like to show data relating to the speedup of our RNN as a result of different parallelization and optimization techniques that we tried over the course of this project.

## Preliminary Results

We have found that using cuBLAS to do many of the matrix operations (multiplication, copying, etc) is faster and more convenient, than parallelizing these operations ourselves. We plan on testing to see if there are other optimizations we can do to improve even further on cuBLAS, but for now we believe that cuBLAS will remain an integral part of our code.

## Issues

I believe the main issue is that we do not entirely know how difficult it will be to turn our basic RNN implementation into the Encoder Decoder network. We initially thought that this would be very difficult, but now we believe that it will be easier than we thought. Despite our wavering opinions, we are not exactly sure how we plan on parallelizing this step, so this should prove to be our biggest challenge going forward.

Also, we do not know how to debug the cuda code efficiently. We tried to use gdb and cuda-gdb, but these command line tools were not very user-friendly. We noticed that there are some remote debugging options in cuda, but we are not sure whether these will be user-friendly or even feasible for this assignment.  


Finally, the backpropagation algorithm is also complicated when we are dealing with matrix derivatives. We are not sure whether we should stick to the cuBLAS for backward computation or use a more naive method instead.

# Schedule

**By** | **Goal** | **Status** |
---| ---| ---|
**November 10th** | Finish proposal, start writing Cuda code for a feed forward network | Completed |
**November 17th** | Continue writing and testing initial implementation | Completed |
**November 19th** | Project Milestone | Completed |
**November 24th** | Continue writing and testing initial implementation|  In-progress |
**November 28th** | Finish initial code |  |
**December 1st**  | Start writing code for an RNN  |  |
**December 5th** | Finish writing RNN code |  |
**December 8th** | Write the Encoder-Decoder RNN and test |  |
**December 12th** | Attempt our additional goals |  |
**December 15th**  | Complete final version of code |  |
