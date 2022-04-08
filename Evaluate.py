# import tensorflow as tf
# import process_data
# import numpy as np
# import cv2
# from pathlib import Path
# from PIL import Image
#
# _, _, (x_test, y_test) = process_data.get_data()
# # y_test = process_data.one_hot_encode(y_test)
# model = tf.keras.models.load_model('classify_dog_cat.h5')
# # model.evaluate(x_test, y_test)
#
# for f in Path('test1').rglob('*.[jp][pn]*'):
#     im = Image.open(f.as_posix())
#     im = np.array(im)
#     cv2.namedWindow('im', cv2.WINDOW_NORMAL)
#     cv2.imshow('im', im)
#     cv2.waitKey()
#     im = cv2.resize(im, (64, 64))
#     im = im / 255.0
#     x_news = np.array([im])
#     y_predict = model.predict(x_news)
#     print(y_predict)
#     if y_predict[0][1] >= 0.5:
#         print('Con chó')
#     else:
#         print('Con mèo')
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras_preprocessing.image import load_img

# from ML.OmOmega.DogCat_Recognition.model import model
from ML.DogCat_Recognition.process_data import Image_Size
from ML.DogCat_Recognition.training import test_generator, batch_size, train_generator

test_filenames = os.listdir("./dogs-vs-cats/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples / batch_size))
test_df['category'] = np.argmax(predict, axis=-1)

label_map = dict((v, k) for k, v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("./dogs-vs-cats/test1/" + filename, target_size=Image_Size)
    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()

results = {
    0: 'cat',
    1: 'dog'
}
im = Image.open("dog_cat/test1/2.jpg")
im = im.resize(Image_Size)
im = np.expand_dims(im, axis=0)
im = np.array(im)
im = im / 255
pred = model.predict_classes([im])[0]
print(pred, results[pred])
