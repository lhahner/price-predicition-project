  
# Defne CNN: Predicting Airbnb Prices from Images

import pandas as pd  
import os
import requests
from PIL import Image
from io import BytesIO  
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Step 1: Load and clean CSV
df = pd.read_csv("listings.csv")
if 'picture_url' not in df.columns or 'price' not in df.columns:
    raise ValueError("Dataset must contain 'picture_url' and 'price' columns.")
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Step 2: Download images
output_dir = "downloaded_images"
os.makedirs(output_dir, exist_ok=True)
image_paths = []

for i, url in enumerate(df['picture_url']):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            path = os.path.join(output_dir, f"image_{i}.jpg")
            img.save(path)
            image_paths.append(path)
        else:
            image_paths.append(None)
    except:
        image_paths.append(None)

df['image_path'] = image_paths
df = df.dropna(subset=['image_path', 'price'])

# Step 3: Load and preprocess images
def load_images_and_prices(df, img_size):
    X, y = [], []
    for _, row in df.iterrows():
        try:
            img = load_img(row['image_path'], target_size=img_size)
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(row['price'])
        except:
            continue
    return np.array(X), np.array(y)

img_size = (128, 128)
X, y = load_images_and_prices(df, img_size)

# Step 4: Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build CNN model
input_layer = Input(shape=(img_size[0], img_size[1], 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='linear')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Step 6: Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Step 7: Predict on validation set
y_pred = model.predict(X_val).flatten()

# Step 8: Scatter plot (Actual vs Predicted)
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.5, color='teal')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')  # ideal line
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted Airbnb Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 9: Histogram of predicted prices
plt.figure(figsize=(8, 5))
plt.hist(y_pred, bins=50, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel("Predicted Price ($)")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Airbnb Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
