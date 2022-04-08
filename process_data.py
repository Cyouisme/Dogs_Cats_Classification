import pandas as pd
import os

# Ảnh càng nhỏ thì model train càng nhanh, ít params hơn nhưng lại ít feature hơn
Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

filenames = os.listdir("dog_cat/train")

categories = []
for f_name in filenames:
    category = f_name.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

