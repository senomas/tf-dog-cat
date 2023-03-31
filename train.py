import os

from tensorflow import keras
from tensorflow import train
from keras.callbacks import ModelCheckpoint
from keras import layers
from keras import preprocessing
from tensorflow.keras import optimizers

image_w = 200
image_h = 200

# define cnn model
def define_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_w, image_h, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile model
    opt = optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

checkpoint_path = "training_checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = train.latest_checkpoint(checkpoint_dir)
if latest is not None:
    print(f'LOAD CHECKPOINT {latest}')
    model.load_weights(latest)

print(f'MODEL INIT {model.summary()}')

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# create data generator
datagen = preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

# prepare iterators
print('training...')
train_it = datagen.flow_from_directory('data/train-sample/', class_mode='binary', batch_size=64, target_size=(image_w, image_h))
print('testing...')
test_it = datagen.flow_from_directory('data/test-sample/', class_mode='binary', batch_size=64, target_size=(image_w, image_h))

# fit model
print('model.fit')
history = model.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=40, callbacks=[cp_callback], verbose=2)
print(f'MODEL SUMMARY {model.summary()}')

# evaluate model
print('model.evaluate')
_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))