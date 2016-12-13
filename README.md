This is the code used in the paper "Modeling cognitive deficits following neurodegenerative diseases and traumatic brain
injuries with deep convolutional neural networks" by Bethany Lusch, Jake Weholt, Pedro D. Maia, and J. Nathan Kutz.

The code is written by Bethany Lusch and Jake Weholt.

The Python code uses TensorFlow and damages a CNN trained on the MNIST dataset. To set up TensorFlow, download the MNIST dataset, and train the same CNN on it, see instructions in the TensorFlow tutorial "Deep MNIST for Experts." Then you will be ready to use our code to damage the network.

The Matlab code uses Matconvnet. We use two pre-trained CNNs that are provided on http://www.vlfeat.org/matconvnet/pretrained/. First, for object classification, we use imagenet-vgg-f on data from ILSVRC 2012. See image-net.org to download the data. Then, for face verification, we use VGG-Face on images from Labeled Faces in the Wild (LFW). See http://vis-www.cs.umass.edu/lfw/ to download the data. 

Our Python and Matlab code follow each other closely. All of the functions for the Python code are in exp_helpers.py. We used the same function names in Matlab, but each function is in a separate .m file. 

Each experiment calls the function base_experiment but passes in different parameters. To rerun our experiments, run the files that start with "exp".

Here is a listing of which experiments were used for each figure:
- Figure 3: exp1.py and exp1end.py
- Figures 4 & 5: exp1_peppers.m
- Figure 6: exp1_faces_prez.m
- Figure 7: exp1.py
- Figure 8: exp2.py, exp2.m, and exp2_faces.m
- Figure 9: exp2.py, exp3.py, and exp4.py
- Figure 10: exp2a.py, exp5.py, exp6.py, and exp7.py
- Figure 11: exp8.py and exp9.py
