import os

from tensorflow.keras import Sequential
from tensorflow.train import latest_checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

checkpoint_path = "training_checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = latest_checkpoint(checkpoint_dir)
if latest is not None:
    print(f'LOAD CHECKPOINT {latest}')
    model.load_weights(latest)

print(f'MODEL SUMMARY {model.summary()}')

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)

# prepare iterators
print('training...')
train_it = datagen.flow_from_directory('data/train-sample/', class_mode='binary', batch_size=64, target_size=(200, 200))
print('testing...')
test_it = datagen.flow_from_directory('data/test-sample/', class_mode='binary', batch_size=64, target_size=(200, 200))

# fit model
print('model.fit')
history = model.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=40, callbacks=[cp_callback], verbose=2)

# evaluate model
print('model.evaluate')
_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))