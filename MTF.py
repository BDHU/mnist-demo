import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#import array [784, 55000], each 784 array is a single piece of 
#training data (there are 55,000) pieces of training data
matrixOfTestData = mnist.train.images

