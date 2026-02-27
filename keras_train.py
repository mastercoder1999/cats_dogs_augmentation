import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "dataset/split/train"
VAL_DIR   = "dataset/split/val"
TEST_DIR  = "dataset/split/test"


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

eval_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = eval_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_gen = eval_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)


test_loss, test_acc = model.evaluate(test_gen)
print(f"\nTEST accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

print("\nClass indices:", train_gen.class_indices)  # e.g. {'cats': 0, 'dogs': 1}

os.makedirs("runs/keras", exist_ok=True)
model.save("runs/keras/cats_dogs_cnn64.keras")
