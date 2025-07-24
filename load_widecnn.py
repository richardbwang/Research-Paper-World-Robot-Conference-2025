import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf  # For tf.data
import pandas as pd
import os
import keras
from keras import layers,Model,Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

model = load_model("regression_model_wide.h5")

# --- Config ---
IMAGE_DIR = "/"
CSV_PATH = "./train.filenamem"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- Load and preprocess DataFrame ---
df = pd.read_csv(CSV_PATH)
df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

labels = df['label'].values.astype('float32')
prev_1 = np.roll(labels, 1)
prev_2 = np.roll(labels, 2)
prev_3 = np.roll(labels, 3)
prev_1[:1] = 0.0
prev_2[:2] = 0.0
prev_3[:3] = 0.0
df['prev_1'] = prev_1
df['prev_2'] = prev_2
df['prev_3'] = prev_3

df = df.iloc[3:].reset_index(drop=True)

# --- Train/test split ---
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# --- TF Dataset loader ---
def load_image(path, label, prev_1, prev_2, prev_3):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    prev_labels = tf.stack([prev_1, prev_2, prev_3])
    return (image, prev_labels), label

def make_dataset(df):
    ds = tf.data.Dataset.from_tensor_slices((
        df['filepath'].values,
        df['label'].values.astype('float32'),
        df['prev_1'].values.astype('float32'),
        df['prev_2'].values.astype('float32'),
        df['prev_3'].values.astype('float32'),
    ))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df)
test_ds = make_dataset(test_df)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# --- Early stopping ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# --- Training ---
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5,
    callbacks=[early_stop]
)

# --- Save & Evaluate ---
#model.save("regression_model_wide10.h5")

loss, mae = model.evaluate(test_ds)
print(f"\nTest Mean Absolute Error: {mae:.2f}")
