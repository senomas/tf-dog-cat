import os

import tensorflow as tf
from tensorflow import keras
from tensorflow import train
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras import optimizers

image_w = 200
image_h = 200

# define cnn model
def define_model():
    model = keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    # compile model
    model.compile(
        optimizer='adam',
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']) 
    return model

model = define_model()

checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = train.latest_checkpoint(checkpoint_dir)
if latest is not None:
    print(f'LOAD CHECKPOINT {latest}')
    model.load_weights(latest)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# prepare datasource
train_ds = preprocessing.image_dataset_from_directory(
    'data/train-sample/',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(image_w, image_h),
    batch_size=32
)

val_ds = preprocessing.image_dataset_from_directory(
    'data/test-sample/',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(image_w, image_h),
    batch_size=32
)

# fit model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=[cp_callback], verbose=2
)
print(f'MODEL SUMMARY {model.summary()}')
model.save('model/model_1')

# evaluate model
_, acc = model.evaluate(val_ds, verbose=0)
print('> %.3f' % (acc * 100.0))