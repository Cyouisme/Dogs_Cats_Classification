from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import BackupAndRestore, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from ML.OmOmega.DogCat_Recognition.model import build_model
from ML.OmOmega.DogCat_Recognition.process_data import df, Image_Size

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
best_checkpoint = ModelCheckpoint(filepath='dog_cat/weight/best_weight.hdf5', verbose=1, save_weights_only=True, save_best_only=True)
backup = BackupAndRestore('dog_cat/backup')
callbacks = [best_checkpoint, earlystop, learning_rate_reduction, backup]


df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
train_df, validate_df = train_test_split(df, test_size=0.20,
                                         random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

train_datagen = ImageDataGenerator(rotation_range=15,
                                   rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1
                                   )

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    "dog_cat/train/", x_col='filename', y_col='category',
                                                    target_size=Image_Size,
                                                    class_mode='categorical',
                                                    batch_size=batch_size)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "dog_cat/train/",
    x_col='filename',
    y_col='category',
    target_size=Image_Size,
    class_mode='categorical',
    batch_size=batch_size
)

test_datagen = ImageDataGenerator(rotation_range=15,
                                  rescale=1. / 255,
                                  shear_range=0.1,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1)

test_generator = train_datagen.flow_from_dataframe(train_df,
                                                   "dog_cat/test1/", x_col='filename', y_col='category',
                                                   target_size=Image_Size,
                                                   class_mode='categorical',
                                                   batch_size=batch_size)

model = build_model()
model.summary()
# optimizer = Adam(lr=0.00146, clipnorm=1.0)
model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["accuracy"])

epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size,
    steps_per_epoch=total_train // batch_size,
    callbacks=callbacks
)
model.save("model1_catsVSdogs_10epoch.h5")
