
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import itertools

# Dummy data (replace with your actual images and prices)
num_samples = 200
img_size = (128, 128)
X = np.random.rand(num_samples, img_size[0], img_size[1], 3)
y = np.random.uniform(50, 500, num_samples)

# Split into train, validation, and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Hyperparameters to tune
param_grid = {
    'learning_rate': [0.001, 0.0005],
    'dropout_rate': [0.3, 0.5],
    'dense_units': [64, 128]
}
combinations = list(itertools.product(*param_grid.values()))
results = []

# Grid search
for lr, dr, units in combinations:
    tf.keras.backend.clear_session()

    input_layer = Input(shape=(img_size[0], img_size[1], 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(units, activation='relu')(x)
    x = Dropout(dr)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0, validation_data=(X_val, y_val))

    val_pred = model.predict(X_val).flatten()
    test_pred = model.predict(X_test).flatten()

    results.append({
        'learning_rate': lr,
        'dropout_rate': dr,
        'dense_units': units,
        'val_mae': mean_absolute_error(y_val, val_pred),
        'val_mse': mean_squared_error(y_val, val_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'test_mse': mean_squared_error(y_test, test_pred),
    })

# Display results
df_results = pd.DataFrame(results)
print(df_results.sort_values(by="val_mae"))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_price_distribution(prices):
    """
    Plots the distribution of listing prices.

    Parameters:
    - prices (array-like): List or array of prices

    Returns:
    - A histogram with KDE showing price distribution
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(prices, bins=30, kde=True, color='skyblue')
    plt.title('Price Distribution of Listings')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_sample_images(images, prices, num_samples=6):
    """
    Displays a grid of sample images along with their price.

    Parameters:
    - images (ndarray): Array of images in shape (n, height, width, 3)
    - prices (array-like): Corresponding prices
    - num_samples (int): Number of samples to show

    Returns:
    - A matplotlib figure of sample images with prices as titles
    """
    plt.figure(figsize=(15, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(f"${prices[i]:.0f}")
        plt.axis('off')
    plt.suptitle("Sample Airbnb Listings with Prices", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plots a scatterplot of actual vs predicted prices.

    Parameters:
    - y_true (array-like): Actual prices
    - y_pred (array-like): Predicted prices

    Returns:
    - A scatter plot showing prediction performance
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, c='orange', edgecolor='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Ideal line
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# plot_price_distribution(y_val)
# show_sample_images(X_val, y_val)
# plot_actual_vs_predicted(y_val, y_pred)   

