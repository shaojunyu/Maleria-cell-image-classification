from PIL import Image
import numpy as np
import cv2
import glob
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

## for storing training information
import json
TRAIN_VAL_HISTORY = "train_val_history.json"

## for plotting
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import plot_malaria_cells  ## my own plotting function

def load_image(image_file):
    image = cv2.imread(image_file)
    image = Image.fromarray(image)
    image = image.resize((50, 50))
    image = np.array(image)
    return image

## TODO: fixed path to task folder
CELL_IMG_DIR = "/labs/colab/BMI500-Fall2020/cell_images"
EPOCHS = 100

if os.path.exists("Cells.npy") and os.path.exists("Labels.npy"):
    cells = np.load("Cells.npy")
    labels = np.load("Labels.npy")
else:
    ## browse images and add labels
    Parasitized = [load_image(p) for p in glob.glob(CELL_IMG_DIR +"/Parasitized/*.png")]
    Uninfected = [load_image(p) for p in glob.glob(CELL_IMG_DIR + "/Uninfected/*.png")]
    cells = np.array(Parasitized + Uninfected)
    labels = np.array([1] * len(Parasitized) + [0]*len(Uninfected)) # uninfected:0, infected: 1
    
    np.save("Cells", cells)
    np.save("Labels", labels)

# normalize  data
cells = cells / 255
# split data into train and test 9:1
X_train, X_test, y_train, y_test = train_test_split(cells, labels, test_size=0.1, random_state=42)
# one-hot encoding of y
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# build CNN model: 2 convolutional layers and 2 dense layers, Dropout layers
model = Sequential([
    Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(50,50,3)),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"),
    MaxPooling2D(pool_size=2),
    Dropout(0.25),
    Flatten(),
    Dense(500,activation="relu"),
    Dropout(0.25),
    Dense(2,activation="softmax")
])

print("The summary of the CNN model is:")
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the model with 1/9 of the data as validation dataset, and record to history object
fit_result = model.fit(X_train, y_train, batch_size=64, epochs=EPOCHS, verbose=2, validation_split=0.125)
with open(TRAIN_VAL_HISTORY, 'w') as fp:
    json.dump(fit_result.history, fp)

# loss, accuracy
results = model.evaluate(X_test, y_test, verbose=1)
print("Loss and Accuracy:", results)

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Confusion matrix:", confusion_matrix(y_true, y_pred))
print("f1 score is: ", f1_score(y_true, y_pred))

## save model
model.save("cells.h5")

## plotting
plot_malaria_cells.plot_loss(TRAIN_VAL_HISTORY, EPOCHS)

## plot out AUROC
# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
## Compute micro-average ROC curve and ROC area
#fpr["malaria"], tpr["malaria"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#plt.figure()
#lw = 2
#plt.plot(fpr[2], tpr[2], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()



## Transfer learning with VGG16
from keras.applications.vgg16 import VGG16
from keras.models import Model

# load model without classifier layers
model = VGG16(include_top=False, input_shape=(50, 50, 3))
# for layer in model.layers:
#     layer.trainable = False

# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(512, activation='relu')(flat1)
output = Dense(2, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.125)

model.evaluate(X_test, y_test)

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_matrix(y_true, y_pred)

f1_score(y_true, y_pred)
