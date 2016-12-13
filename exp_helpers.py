from tensorflow.examples.tutorials.mnist import input_data
import datetime
import tensorflow as tf
import os
import copy
import math
import random
import sys
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

# facilitates the damaging of the network.
def damage_network(network_matrices, dmg_size, pie_chart, coeff, sigma, net_size, high_weight):
    matrix_shapes = get_matrix_shapes(network_matrices)
    matrices_as_vector = vectorize_network(network_matrices)
    damage_indices = get_damage_indices(matrices_as_vector, dmg_size, net_size)
    [matrices_as_vector[damage_indices], num_damaged] = damagefn(matrices_as_vector[damage_indices], pie_chart, high_weight, coeff, sigma)

    # return number damaged including transmission 
    return [reshape_matrices(matrices_as_vector, matrix_shapes), len(damage_indices)]

def damagefn(weights_to_damage, pie_chart, high_weight, coeff, sigma):
    # Red (blockage), Orange (Reflection), Yellow (Filtering) and Green (Transmision)
    num_weights = len(weights_to_damage)
    num_types = (np.array(pie_chart) != 0).sum()
    if num_types > 1:
        # randomly split weights into four groups
        permuted_ind = np.random.permutation(num_weights)
        
        end_blocked = np.round(pie_chart[0]*num_weights)
        blocked_ind = permuted_ind[0:end_blocked]

        end_reflected = np.round(pie_chart[1]*num_weights) + end_blocked
        reflected_ind = permuted_ind[end_blocked:end_reflected]

        end_filtered = np.round(pie_chart[2]*num_weights) + end_reflected
        filtered_ind = permuted_ind[end_reflected:end_filtered]

        weights_to_damage[blocked_ind] = 0
        weights_to_damage[reflected_ind] = .5 * weights_to_damage[reflected_ind]
        weights_to_damage[filtered_ind] = weight_filter(weights_to_damage[filtered_ind], high_weight, coeff, sigma)
        
        num_damaged = len(blocked_ind) + len(reflected_ind) + len(filtered_ind)
    else:
        # if they are all of same type, this saves a lot of time: randperm is
        # slow
        if pie_chart[0] == 1:
            # blockage only: set all to 0
            num_damaged = len(weights_to_damage)
            weights_to_damage = 0
        elif pie_chart[1] == 1:
            # reflection only: halve all weights
            weights_to_damage = .5 * weights_to_damage
            num_damaged = len(weights_to_damage)
        elif pie_chart[2] == 1:
            # filtering only
            weights_to_damage = weight_filter(weights_to_damage, high_weight, coeff, sigma)
            num_damaged = len(weights_to_damage)
        # final case is transmission only, so do no damage 
    return [weights_to_damage, num_damaged]

def weight_filter(weights_to_damage, high_weight, coeff, sigma):
    if len(weights_to_damage):
        scaled_weights = weights_to_damage / high_weight # mostly between -1 and 1
        signs = np.sign(scaled_weights)
        filtered_weights = signs * filter_polynomial(np.abs(scaled_weights), coeff, sigma) * high_weight
        return filtered_weights
    else:
        # this if-else block might save some time in the common case of 
        # having empty weights vector 
        return weights_to_damage
    
def filter_polynomial(x, coeff, sigma):
    y = coeff[0] * x**2 + coeff[1] * x + coeff[2] + sigma * np.random.randn(len(x),)

    
    return y
    
# filter network:
# filter_type = "inside", filters inside-out.
# filter_type = "outside", filters outside-in.
# filters from median + and - the percentile window size, so the window is actually 2*percentile_window in size.
def filter_network(network_matrices, percentile_window, damage_amt, filter_type, net_size):
    matrix_shapes = get_matrix_shapes(network_matrices)
    matrices_as_vector = vectorize_network(network_matrices)
    if filter_type == "inside":
        [return_vector, num_damaged] = filter_vector_in(matrices_as_vector, percentile_window, damage_amt, net_size)
    elif filter_type == "outside":
        [return_vector, num_damaged] = filter_vector_out(matrices_as_vector, percentile_window, damage_amt, net_size)
    return [reshape_matrices(return_vector, matrix_shapes), num_damaged]
    
def sparsify_network(network_matrices, percentile_window, net_size):
    # sparsify network by removing small weights (within percentile_window)
    damage_amt = 0
    filter_type = "inside"
    [sparsified_networks, num_removed] = filter_network(network_matrices, percentile_window, damage_amt, filter_type, net_size)
    return [sparsified_networks, num_removed]

    
# returns shapes of original network matrices for reshaping
def get_matrix_shapes(network_matrices):
    list_of_shapes = []
    for matrix in network_matrices:
        list_of_shapes.append(list(matrix.shape))
    return list_of_shapes

# turns network matrices into one long vector
def vectorize_network(network_matrices):
    vector = np.empty(0)
    for matrix in network_matrices:
        vector = np.append(vector, np.reshape(copy.copy(matrix), -1))
    return vector

# returns random sample of indices to damage
def get_damage_indices(matrices_as_vector, dmg_size, net_size):
    num_elements_to_damage = int(math.floor(dmg_size * net_size))
    non_zero_elements = np.nonzero(matrices_as_vector)
    linear_indices = random.sample(range(0, len(non_zero_elements[0])), min(num_elements_to_damage, len(non_zero_elements[0])))
    return non_zero_elements[0][linear_indices]

# reshapes damaged vector into original network matrices
def reshape_matrices(matrix_as_vector, matrix_shapes):
    matrices = []
    vector_lengths = get_vector_lengths(matrix_shapes)
    for i in range(len(matrix_shapes)):
        matrices.append(\
              np.reshape(\
                matrix_as_vector[sum(vector_lengths[0:i+1]):sum(vector_lengths[0:(i+2)])],\
                       matrix_shapes[i]))
    return matrices

def get_vector_lengths(matrix_shapes):
    length = [0]
    for shape in matrix_shapes:
        length.append(np.prod(shape))
    return length

# helper function for filter, inside damage
def filter_vector_in(matrices_as_vector, percentile_window, damage_amt, net_size):
    vec_size = len(matrices_as_vector)
    if vec_size > net_size:
        # means that we already set some thresholded section to 0. Don't want to count those zeros
        # in calculating percentile
        num_thresholded = vec_size - net_size
        print(num_thresholded)
        sorted_ind = np.argsort(matrices_as_vector) #indices that would sort matrices_as_vector
        sorted_ind = sorted_ind[num_thresholded:vec_size] # cut off num_thresholded from front
        values_to_check = matrices_as_vector[sorted_ind]
    else:
        values_to_check = matrices_as_vector
    upper_perc = np.percentile(values_to_check, 50 + percentile_window)
    print(upper_perc)
    lower_perc = np.percentile(values_to_check, 50 - percentile_window)
    print(lower_perc)
    damaged_number = 0
    for i in range(len(matrices_as_vector)):
        if (matrices_as_vector[i] <= upper_perc and matrices_as_vector[i] >= lower_perc):
            matrices_as_vector[i] = damage_amt
            damaged_number = damaged_number + 1
    return [matrices_as_vector, damaged_number]

# helper function for filter, outside damage
def filter_vector_out(matrices_as_vector, percentile_window, damage_amt, net_size):
    vec_size = len(matrices_as_vector)
    if vec_size > net_size:
        # means that we already set some thresholded section to 0. Don't want to count those zeros
        # in calculating percentile
        num_thresholded = vec_size - net_size
        print(num_thresholded)
        sorted_ind = np.argsort(np.abs(matrices_as_vector))
        sorted_ind = sorted_ind[num_thresholded:vec_size]
        values_to_check = matrices_as_vector[sorted_ind]
    else:
        values_to_check = matrices_as_vector
    upper_perc = np.percentile(values_to_check, 100 - percentile_window)
    print(upper_perc)
    lower_perc = np.percentile(values_to_check, 0 + percentile_window)
    print(lower_perc)
    damaged_number = 0
    for i in range(len(matrices_as_vector)):
        if (matrices_as_vector[i] > upper_perc or matrices_as_vector[i] < lower_perc):
            matrices_as_vector[i] = damage_amt
            damaged_number = damaged_number + 1
    return [matrices_as_vector, damaged_number]
    
# returns final output values for every class by image.
def get_output_class_vectors(network_matrices, sess, y_conv, x, test_images, keep_prob, W_conv1, W_conv2, W_fc1):
    return sess.run(y_conv, feed_dict={x: test_images, 
                               keep_prob: 1.0,
                                 W_conv1: network_matrices[0],
                                 W_conv2: network_matrices[1],
                                   W_fc1: network_matrices[2]})

# returns labels predicted by the network
def get_predicted_labels(predicted_vectors):
    return np.argmax(predicted_vectors, axis=1)
    
# returns accuracy of the network
def get_network_accuracy(actual_labels, predicted_labels):
    errors = np.subtract(actual_labels, predicted_labels)
    errors[np.nonzero(errors)] = 1
    return 1 - float(sum(errors))/float(len(errors))

# handles printing everything to .csv file. 
def output_data_to_csv(file_name, damage_size, trial_number, actual_labels, predicted_labels, class_scores):
    indices = range(len(actual_labels))
    fd = open(file_name, 'a')
    is_wrong = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] - predicted_labels[i] == 0:
            is_wrong = 0
        else:
            is_wrong = 1
        fd.write('%d,%f,%f,%d,%d,%d,' % (i, damage_size, trial_number, actual_labels[i], predicted_labels[i], is_wrong))
        for class_score in class_scores[i]:
            fd.write('%f,' % class_score) 
        fd.write('\n')
    fd.close

# handles printing everything to .csv file. 
def output_summary_data_to_csv(file_name, accuracies, trial_counter):
    np.savetxt(file_name, accuracies, delimiter=",", fmt='%1.4f')


    
# returns new and unique file name
def get_file_name(trial_counter, expnum):
    return ("./exp%s/mnist_cnn_exp%s_trial_%d_" % (expnum, expnum, trial_counter)) + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f.csv")

# creates a new .csv file in working directory
def initialize_new_file(header_string, trial_counter, expnum):
    file_name = get_file_name(trial_counter, expnum)
    fd = open(file_name, 'a')
    fd.write(header_string) 
    fd.close()
    return file_name


# Returns actual image labels
def get_actual_image_labels(sess, actual, y_, test_labels):
    return sess.run(actual, feed_dict={y_: test_labels})

# Assembles datasets    
def prepare_data():
    # Create TensorFlow data objects (contains all images and labels)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    ############
    # Even out class sizes, 892 images each, limited by the smallest class of only 892 images.
    test_images = []
    test_labels = []
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    magic_val = 852
    count = 0
    for i in range(len(mnist.test.labels)):
        index = np.nonzero(mnist.test.labels[i])[0][0]
        if counts[index] < magic_val:
            test_images.append(mnist.test.images[i])
            test_labels.append(mnist.test.labels[i])
            count = count + 1
        counts[index] = counts[index] + 1

    # New evenly sized test sets.
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)
    
    return [test_images, test_labels]
    
def setup_network():
    sess = tf.InteractiveSession()
    
    ############
    # TensorFlow setup
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 10])
    b_fc1 = bias_variable([10])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    keep_prob = tf.placeholder("float")
    y_conv = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    predicted = tf.argmax(y_conv, 1)
    actual = tf.argmax(y_, 1)
    correct_prediction = tf.equal(predicted, actual)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())

    ############
    # import already trained model/network from file.
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(os.getcwd())
    saver.restore(sess, ckpt.model_checkpoint_path)

    ############
    # Convert weight matrices from tensorflow tensor objects to real value numpy arrays with sess.run()
    # packs them together in a list to contain all of the weight matrices.
    matrices_to_damage =\
        [np.asarray(sess.run(W_conv1)),
         np.asarray(sess.run(W_conv2)),
         np.asarray(sess.run(W_fc1))]
         
    return [sess, actual, y_, y_conv, x, keep_prob, W_conv1, W_conv2, W_fc1, matrices_to_damage]

def setup_experiment():
    [test_images, test_labels] = prepare_data()
    [sess, actual, y_, y_conv, x, keep_prob, W_conv1, W_conv2, W_fc1, matrices_to_damage] = setup_network()

    # List of actual test image labels for comparison.
    actual_test_image_labels = get_actual_image_labels(sess, actual, y_, test_labels)
    
    return [matrices_to_damage, sess, y_conv, x, test_images, keep_prob, W_conv1, W_conv2, W_fc1, actual_test_image_labels]
    
def base_experiment(expnum = 1, pie_chart = [1, 0, 0, 0], damages_values = np.arange(0,1,0.01), detailed_file_flag = 0, max_trials = float('inf'), histogram_flag = 0, filter_type = None, aging_flag = 0, header_string = "image_index, damage_size, trial, correct_class, inferred_class, is_wrong, pred_0, pred_1" +\
                    ", pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9\n", coeff = [-.2774, .9094, -.0192], sigma = .05, sparsity_cutoff = 0):
                    
    filedir = './exp%s/' % (expnum)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
        
        
    ############
    # User defined model parameters:
    # default_damage_amount = what the weights are set to when they are damaged
    # damages_values = range of values, 0 to 1 in steps of 0.01 to represent network damage amount.
    print(damages_values)
    # Output file header for the top of the csv file.
    # histogram_flag can be 0 for random damage, 1 for chopping parts of the histogram out 
    
    [matrices_to_damage, sess, y_conv, x, test_images, keep_prob, W_conv1, W_conv2, W_fc1, actual_test_image_labels] = setup_experiment()
    vectorized = vectorize_network(matrices_to_damage)
    net_size = len(vectorized)
   
    high_weight = np.percentile(np.abs(vectorized), 95)
    
    if sparsity_cutoff:
        [matrices_to_damage, num_removed] = sparsify_network(matrices_to_damage, sparsity_cutoff, net_size)
        print(net_size)
        net_size = net_size - num_removed
        print(net_size)

    ############
    # Damage and file output loop:
    accuracies = np.zeros((len(damages_values), 3))
    trial_counter = 1
    while True:
        file_name = initialize_new_file(header_string, trial_counter, expnum)
        dmg_counter = 0;
        matrices_to_damage_this_trial = matrices_to_damage
        for dmg_size in damages_values:
            if histogram_flag:
                default_damage_amount = 0
                [damaged_network, num_damaged] = filter_network(matrices_to_damage_this_trial, dmg_size, default_damage_amount, filter_type, net_size)
            else:
                [damaged_network, num_damaged] = damage_network(matrices_to_damage_this_trial, dmg_size, pie_chart, coeff, sigma, net_size, high_weight)
            predicted_vectors = get_output_class_vectors(damaged_network, sess, y_conv, x, test_images, keep_prob, W_conv1, W_conv2, W_fc1)
            predicted_test_image_labels = get_predicted_labels(predicted_vectors)
            network_accuracy = get_network_accuracy(actual_test_image_labels, predicted_test_image_labels)
        
            if detailed_file_flag:
                output_data_to_csv(file_name, dmg_size, trial_counter, actual_test_image_labels, predicted_test_image_labels, predicted_vectors)
            
            if aging_flag:
                # want damage to accumulate over this trial (but not between trials)
                matrices_to_damage_this_trial = damaged_network
            
            accuracies[dmg_counter, 0] = dmg_size
            accuracies[dmg_counter, 1] = num_damaged
            accuracies[dmg_counter, 2] = network_accuracy
            dmg_counter = dmg_counter + 1;
        if not detailed_file_flag:
            output_summary_data_to_csv(file_name, accuracies, trial_counter)
        print("Trials completed: %d\n" % trial_counter)
        trial_counter = trial_counter + 1
        if trial_counter > max_trials:
            break
            