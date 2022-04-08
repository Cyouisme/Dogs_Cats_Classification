from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Input
from tensorflow.keras.models import Model

import process_data


def build_model():
    input_shape = Input((process_data.Image_Width, process_data.Image_Height, process_data.Image_Channels), name='Input')
    x = Conv2D(32, (3, 3), kernel_initializer='he_normal', activation='relu')(input_shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_shape, x)
    return model


if __name__ == '__main__':
    model = build_model()
    model.summary()


# """ Tuỳ vào lượng dữ liệu, model mà có thể chọn số layer"""
# model = Sequential()
#
# model.add(Conv2D(32, (3, 3), activation='relu',
#                  input_shape=(process_data.Image_Width, process_data.Image_Height, process_data.Image_Channels)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# # model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop', metrics=['accuracy'])
#
# # if __name__ == '__main__':
# #     get_model((64, 64, 3))
# model.summary()
