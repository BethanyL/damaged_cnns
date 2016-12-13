This is the code used in the paper "Modeling cognitive deficits following neurodegenerative diseases and traumatic brain
injuries with deep convolutional neural networks" by Bethany Lusch, Jake Weholt, Pedro D. Maia, and J. Nathan Kutz.

The code is written by Bethany Lusch and Jake Weholt.

The Python code uses TensorFlow and damages a CNN trained on the MNIST dataset. To set up TensorFlow, download the MNIST dataset, and train the same CNN on it, see instructions in the TensorFlow tutorial "Deep MNIST for Experts." Then you will be ready to use our code to damage the network.

The Matlab code uses Matconvnet. We use two pre-trained CNNs that are provided on http://www.vlfeat.org/matconvnet/pretrained/. First, for object classification, we use imagenet-vgg-f on data from ILSVRC 2012. See image-net.org to download the data. Then, for face verification, we use VGG-Face on images from Labeled Faces in the Wild (LFW). See http://vis-www.cs.umass.edu/lfw/ to download the data. 

