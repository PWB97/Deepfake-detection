import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks

import time

from meso.meso import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
start = time.time()

img_width, img_height = 64, 64
batch_size = 2000
epochs = 20

train_data_path = ''
validation_data_path = ''

# model = Meso4().model
model = MesoInception4().model

train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    shuffle=True)

target_dir = './meso/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save('./meso/model.h5')
model.save_weights('./meso/weights.h5')

# Calculate execution time
end = time.time()
dur = end - start

if dur < 60:
    print("Execution Time:", dur, "seconds")
elif dur > 60 and dur < 3600:
    dur = dur / 60
    print("Execution Time:", dur, "minutes")
else:
    dur = dur / (60 * 60)
    print("Execution Time:", dur, "hours")
