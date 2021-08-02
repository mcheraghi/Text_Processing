import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns   #to have pair plot


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)



# -----Get the data from the archive
#LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
   

print(train_images[10:100,25,23])
print(train_images[:500])

for i in range(10):
	plt.imshow(train_images[i])
	plt.title(str(i)+str(train_images[i].shape))
	plt.pause(0.3)
	
                    




# -----Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


print(model.summary())

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())
# -----Train the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=2, 
                    validation_data=(test_images, test_labels))


# -----Model evaluation
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)


# -----Model prediction
for i in range(10):
	img_array = keras.preprocessing.image.img_to_array(test_images[i])
	img_array = tf.expand_dims(img_array, 0) # Create a batch
	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	print(score, test_labels[i])
	print(
	    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)), 
	    "    Real category is {}".format(class_names[int(test_labels[i])])
	)
	


