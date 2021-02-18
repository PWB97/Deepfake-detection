import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve

from tqdm import tqdm

from meso.meso import *

validation_data_path = '/home/asus/ffdf_40/test'

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'

img_width, img_height = 64, 64
batch_size = 2000
epochs = 20

frame_y_gd = []
frame_y_pred = []

model = MesoInception4()
# model = Meso4()
model.load('/home/asus/Code/checkpoint/ff/msin/weights.h5')

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
pred_pro = pred

pred = np.rint(pred)
f_fpr, f_tpr, _ = roc_curve(gd, pred_pro)
test_frame_acc = accuracy_score(gd, pred)
test_frame_auc = roc_auc_score(gd, pred_pro)
test_frame_f1 = f1_score(gd, pred)
test_frame_pre = precision_score(gd, pred)
test_frame_recall = recall_score(gd, pred)

np.save('./m/msi/f_fpr.npy', f_fpr)
np.save('./m/msi/f_tpr.npy', f_tpr)

print('acc:, auc:, f1_score, precision_score, recall_score')
print(test_frame_acc, test_frame_auc, test_frame_f1, test_frame_pre, test_frame_recall)