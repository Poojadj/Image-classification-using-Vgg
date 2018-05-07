from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras.constraints import maxnorm


# dimensions of our images.
img_width, img_height = 28,28

train_data_dir = 'E:\Pro\Lugg'
#validation_data_dir = 'data/validation'
nb_train_samples = 572
#nb_validation_samples = 800
epochs = 2
batch_size = 128

#if K.image_data_format() == 'channels_first':
 #   input_shape = (3, img_width, img_height)
#else:
#  input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', border_mode='same',
                 #input_shape=input_shape))


model.add(Conv2D(32, (3, 3), activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), activation='relu', border_mode='same'))
model.add(Conv2D(64, (3, 3), activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', border_mode='same'))
model.add(Conv2D(128, (3, 3), activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024, activation='relu',W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',W_constraint=maxnorm(3)))
model.add(Dropout(0.5))


model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#validation_generator = test_datagen.flow_from_directory(
 #  validation_data_dir,
  # target_size=(img_width, img_height),
   # batch_size=batch_size,
    #class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs )
  #  validation_data=validation_generator,
   # validation_steps=nb_validation_samples // batch_size)
model.save_weights('try.h5')
