import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pathlib import Path
from decepticons import decepticons

class decepticons_regressor:

    def __init__(self):
        self.regressor = LinearRegression()
        self.fitted = True
        self.model_path = Path("./checkpoints/linear_regressor.pkl")

    def prepare_data_labels(self, train_df, val_df, test_df):
        train_labels = train_df['price'].replace('[\$,]', '', regex=True).astype(float)
        val_labels = val_df['price'].replace('[\$,]', '', regex=True).astype(float)
        return train_labels, val_labels

    def train_regressor(self, train_scores, train_prices):
        if self.fitted == False:
            self.regressor.fit(train_scores.reshape(-1, 1), train_prices)
            self.fitted = True
            print("Linear regression model trained on (true review score → price)")
        else:
            print("Linear regression model already trained")
            return

    def predict_prices(self, review_scores):
        if isinstance(review_scores, pd.Series):
            review_scores = review_scores.values
        if review_scores.ndim == 1:
            review_scores = review_scores.reshape(-1, 1)
        return self.regressor.predict(review_scores)

    def evaluate(self, predicted_prices, actual_prices):
        mse = mean_squared_error(actual_prices, predicted_prices)
        print(f"Mean Squared Error on predicted prices: {mse:.2f}")
        return mse

    def save_model(self):
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.regressor, self.model_path)
            print(f"Linear regression model saved to: {self.model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load_model(self):
        print("Loading Linear Regression model")
        try:
            self.regressor = joblib.load(self.model_path)
            self.fitted = True
            print("Linear regression model loaded successfully.")
        except FileNotFoundError:
            print(f"Model file not found at: {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")

if __name__ == "__main__":
    decepticons_reg = decepticons_regressor()
    dec = decepticons()

    decepticons_reg.load_model()
    train_df, val_df, test_df = dec.split_data()

    train_labels, val_labels = decepticons_reg.prepare_data_labels(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    decepticons_reg.train_regressor(
        train_scores=train_df['review_scores_value'].values.reshape(-1, 1),
        train_prices=train_labels,
    )

    predicted_prices = decepticons_reg.predict_prices(
        review_scores=val_df['review_scores_value'].values.reshape(-1, 1)
    )

    if decepticons_reg.fitted:
        mse = decepticons_reg.evaluate(
            predicted_prices,
            val_labels
        )
    else:
        raise EOFError

    decepticons_reg.save_model()




