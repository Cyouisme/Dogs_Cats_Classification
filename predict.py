import cv2 as cv
import numpy as np
from pathlib import Path
from PIL import Image
from ML.OmOmega.DogCat_Recognition.model import build_model

model = build_model()
model.load_weights(r"D:\BaoChung\ML\OmOmega\DogCat_Recognition\dog_cat\weight\best_weight.hdf5")

""" Tạo một list đưa hết tất cả các ảnh vào rồi đem đi so sánh với model"""
list_im = []
for f in Path('cat_test').rglob('*.[jp][pn]*'):
    im = Image.open(f.as_posix())
    im = np.array(im)
    img_true = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    img_true = cv.resize(img_true, (128, 128))
    img_true = img_true / 255.0
    img_true = np.expand_dims(img_true, axis=0)  # -expand_dims - thêm chiều dữ liệu
    list_im.append(img_true)

y_predict = model.predict(np.vstack(list_im))  # vstack - ghép tất cả các ảnh lại thành 1 ảnh to
for img, y in zip(list_im, y_predict):
    cv.namedWindow('im', cv.WINDOW_NORMAL)
    cv.imshow('im', np.squeeze(img, axis=0))  # squeeze - giảm chiều của ảnh về ban đầu
    if np.argmax(y) == 1:  # argmax - trả về index của max
        print("That's a Dog!")
    else:
        print("That's a Cat!")
    cv.waitKey()
# 50ms - 95%
