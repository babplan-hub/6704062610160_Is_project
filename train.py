import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# =====================
# CONFIG
# =====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

train_dir = "dataset/train"
val_dir = "dataset/val"

# =====================
# CREATE MODEL FOLDER (if not exist)
# =====================
os.makedirs("model", exist_ok=True)

# =====================
# DATA GENERATOR
# =====================
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# =====================
# SAVE CLASS NAMES (สำคัญมาก)
# =====================
class_names = list(train_gen.class_indices.keys())

with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

print("Saved class names:", class_names)

# =====================
# BUILD MODEL
# =====================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================
# TRAIN
# =====================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# =====================
# SAVE MODEL
# =====================
model.save("model/dog_model.h5")

print("✅ Train and save model successfully!")