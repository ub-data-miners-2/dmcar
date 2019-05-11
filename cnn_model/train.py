# USAGE
# python train.py --dataset images --model traffic_sign.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import re

def load_data(folder, important_label):
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = []
    for label in os.listdir('./'+folder+'/'):
        if re.search('(\.csv|\.txt|\.DS_Store)', label):
            continue

        if label == important_label:
            category = 1
        else:
            category = 0

        for filename in os.listdir('./'+folder+'/'+label+'/'):
            if re.search('(\.csv|\.txt|\.DS_Store)', filename):
                continue
            imagePaths.append(['./'+folder+'/'+label+'/'+filename, category])

    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        # print(imagePath)
        image = cv2.imread(imagePath[0])
        # image = cv2.equalizeHist(image)
        image = cv2.resize(image, (32, 32))
        image = img_to_array(image)
        data.append(image)
        # extract the class label from the image path and update the
        # labels
        listlabel = imagePath[1]
        # label = 1 if listlabel == "stop" else 0
        labels.append(imagePath[1])
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return [data, labels]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epoch", type=int, default="25")
ap.add_argument("-c", "--category", type=str, required=True, help="the category to differentiate from others [stop, yield, railroad, speed]")
ap.add_argument("-m", "--model", help="path to output model")
ap.add_argument("-p", "--plot", type=str, help="path to output loss/accuracy plot")
args =vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate, and batch size
EPOCHS = args["epoch"]
INIT_LR = 5e-3
BS = 32
# initialize the data and labels
print("[INFO] loading images...")
trainX, trainY = load_data('training', args["category"])


# convert the labels from integers to vectors
trainY = to_categorical(trainY,num_classes=2)
testX, testY = load_data('testing', args["category"])
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.3,
    height_shift_range=0.2, shear_range=0.1, zoom_range=0.2, horizontal_flip=True,
    fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=32, height=32, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    shuffle=True,
    use_multiprocessing=True,
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# test
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=["other", args['category']]))

# save the model to disk
print("[INFO] serializing network...")
model_name = args["model"]
if model_name is None:
    model_name = args["category"] + "_not_" + args["category"] + ".model"
model.save(model_name)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on " + args['category'] + "/Not " + args['category'])
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plotname = args["plot"]
if plotname is None:
    plotname = args['category'] + '.png'
plt.savefig(plotname)
