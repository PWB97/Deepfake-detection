import os

from meso import *
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm import tqdm

validation_data_path = '/data2/guesthome/wenbop/ffdf_c40/test'

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'

img_width, img_height = 64, 64
batch_size = 2000
epochs = 20

frame_y_gd = []
frame_y_pred = []

model = Meso4()
model.load('./meso/weights.h5')

test_datagen = ImageDataGenerator(rescale=1. / 255)


validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

i = 0
for X, y in tqdm(validation_generator, desc='Validating'):
    y_ = model.predict(X)
    frame_y_pred += y_.tolist()
    frame_y_gd += y.tolist()
    i += 1
    if i >= 37:
        break


gd = np.array(frame_y_gd)
pred = np.array(frame_y_pred)

pred = np.rint(pred)

test_frame_acc = accuracy_score(gd, pred)
test_frame_auc = roc_auc_score(gd, pred)

print('acc:, auc:')
print(test_frame_acc, test_frame_auc)