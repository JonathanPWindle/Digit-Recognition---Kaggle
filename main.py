import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import modelUtils
import dataPreProcess
import numpy as np
import pandas as pd

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
data = dataPreProcess.preprocess()

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

learningRate = 0.0001
learningEpochs = 20000
displaySteps = 100

# First convolutional layer
# Computes 32 features for a 5x5 patch
# Weight tensor shape = [5,5,1,32]

conv1Weights = modelUtils.weightVariable([5, 5, 1, 32])
conv1Bias = modelUtils.biasVariable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve the x_image with the convolutional weights and add the bias
conv1Output = tf.nn.relu(modelUtils.conv2D(x_image, conv1Weights) + conv1Bias)

# Apply max pooling, reducing the size to 14x14
pooling1Output = modelUtils.maxPool2x2(conv1Output)

# Second convolutional layer
# Computes 64 features for a 5x5 patch
# Weight tensor shape = [5,5,32,64]

conv2Weights = modelUtils.weightVariable([5, 5, 32, 64])
conv2Bias = modelUtils.biasVariable([64])

# Convolve the output from the first conv layer and the second convolutional weights
conv2Output = tf.nn.relu(modelUtils.conv2D(pooling1Output, conv2Weights) + conv2Bias)

# Apply max pooling to outputs
pooling2Output = modelUtils.maxPool2x2(conv2Output)

#
# Fully connected layer
#

fullyCon1Weights = modelUtils.weightVariable([7 * 7 * 64, 1024])
fullyCon1Bias = modelUtils.biasVariable([1024])

pool2OutputsFlattened = tf.reshape(pooling2Output, [-1, 7 * 7 * 64])
fullyConOutput = tf.nn.relu(tf.matmul(pool2OutputsFlattened, fullyCon1Weights) + fullyCon1Bias)

#
# To avoid overfitting, apply dropout before final readout layer.
#

keepProb = tf.placeholder(tf.float32)
fullyConDrop = tf.nn.dropout(fullyConOutput, keepProb)

#
# Readout layer
#

fullyCon2Weights = modelUtils.weightVariable([1024, 10])
fullyCon2Bias = modelUtils.biasVariable([10])

yConv = tf.matmul(fullyConDrop, fullyCon2Weights) + fullyCon2Bias

#
# TRAINING
#

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yConv))
trainStep = tf.train.AdamOptimizer(learningRate).minimize(crossEntropy)
correctPrediction = tf.equal(tf.arg_max(yConv, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

prediction = tf.arg_max(yConv, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(learningEpochs):
        randomNum = np.random.randint(0, 838)
        if i % displaySteps == 0:
            trainAccuracy = accuracy.eval(feed_dict={x: data["xBatch"][randomNum], y: data["yBatch"][randomNum], keepProb: 1.0})
            print("Step %d, training accuracy: %g" % (i, trainAccuracy))
        trainStep.run(feed_dict={x: data["xBatch"][randomNum], y: data["yBatch"][randomNum], keepProb:0.5})

    predictions = []
    for batch in data["testBatches"]:
        predictions.extend(prediction.eval(feed_dict={x: batch, keepProb:1.0}))
    ids = []
    for i in range(1, len(predictions) + 1):
        ids.append(i)
    data = {"ImageId": ids, "Label": predictions}

    df = pd.DataFrame(data=data)
    df.to_csv('Results/results.csv', index=False)
    #print("Test Accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y: mnist.test.labels, keepProb: 1.0}))