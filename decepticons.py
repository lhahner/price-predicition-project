import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
from transformers import BertTokenizerFast, AutoModel
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
"""
General TODO list:
 - Write weights to file and read in [x]
 - Train more than one epoch [x] -> done by runnign the script multiple times
 - Combine with regression head to determination [] -> Pipeline: BERT encodes semantic meaning of the comment -> 
    Predicts review score -> translates sentiment into expected price
 - Test Bert with custom reviews []
 - Visualize [] 
"""
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

class decepticons(nn.Module):

    """
    Load Transformer Model
        - Choose a pre-trained Transformer Model
        - Load tokenizer and model from Hugging Face Transformers
    """
    def __init__(self):
        super(decepticons, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.2)
        self.rectified_linear_unit = nn.ReLU()
        self.linear_layer_1 = nn.Linear(768, 512)
        self.linear_layer_2 = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, 1)  # Final scalar output
        self.listings = pd.read_csv("./data/listings.csv")
        self.reviews = pd.read_csv("./data/reviews.csv")
        self.decepticons_weights_path = Path("./checkpoints/decepticons.pt")
        self.test_mode = True
        self.prediction_mode = False


    """
     Prepare Text Features
        - Load data and preprocess data as for transformers
        - @source https://mccormickml.com/2021/06/29/combining-categorical-numerical-features-with-bert/
    """
    def prepare_features(self):
        # Normalize column names
        self.listings.columns = self.listings.columns.str.strip().str.lower()
        self.reviews.columns = self.reviews.columns.str.strip().str.lower()

        # Merge listings and reviews
        merged = pd.merge(
            self.listings,
            self.reviews,
            left_on="id",
            right_on="listing_id",
            how="left"
        )

        # Rename for clarity
        merged = merged.rename(columns={
            "id_x": "id",
            "id_y": "review_id"
        })

        # Select relevant columns
        df = merged[["id", "price", "review_scores_value","comments"]]

        # Drop rows where any of the selected columns have NaN
        df = df.dropna(subset=["id", "price", "review_scores_value","comments"])

        return df

    """
    Running a forward pass through the model
        - first providing the inputs to the bert model
        - Second running it through the 
    """
    def forward(self, sent_id, mask):
        outputs = self.bert(sent_id, attention_mask=mask, return_dict=True)
        cls_embedding = outputs.pooler_output  # shape [batch_size, 768]

        x = self.linear_layer_1(cls_embedding)  # 768 → 512
        x = self.rectified_linear_unit(x)
        x = self.dropout(x)
        x = self.linear_layer_2(x)  # 512 → 512
        x = self.rectified_linear_unit(x)
        x = self.dropout(x)

        out = self.output_layer(x)  # 512 → 1
        return out.squeeze(-1)  # [batch_size]

    """
    Splits the data into train, validation and test sets.
    Returns:
        train_df, val_df, test_df: DataFrames split accordingly
    """
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):

        df = self.prepare_features()

        # First split off test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Then split train_val into train and val
        val_relative_size = val_size / (1 - test_size)  # adjust val size relative to train_val

        train_df, val_df = train_test_split(
            train_val_df, test_size=val_relative_size, random_state=random_state, shuffle=True
        )

        return train_df, val_df, test_df

    """
    Generate Text Embeddings
        - Tokenize the text data
        - Pass tokenized input through the model to get embeddings
        @Source https://www.geeksforgeeks.org/nlp/fine-tuning-bert-model-for-sentiment-analysis/
    """
    def generate_embeddings(self, train_df=None, test_df=None, val_df=None, batch_size=16):
        # load tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        pad_len = int(train_df["comments"].str.len().mean())

        tokens_train = tokenizer.batch_encode_plus(
            train_df["comments"].tolist(),
            max_length=pad_len,
            padding=True,
            truncation=True
        )

        tokens_test = tokenizer.batch_encode_plus(
            test_df["comments"].tolist(),
            max_length=pad_len,
            padding=True,
            truncation=True
        )

        tokens_val = tokenizer.batch_encode_plus(
            val_df["comments"].tolist(),
            max_length=pad_len,
            padding=True,
            truncation=True
        )

        return tokens_train, tokens_val, tokens_test

    """
    Discretize Classification labels as of review_score_value
    """
    def discretize_labels(self, train_df=None, val_df=None, test_df=None):
        train_labels = pd.qcut(train_df['review_scores_value'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        val_labels = pd.qcut(val_df['review_scores_value'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        test_labels = pd.qcut(test_df['review_scores_value'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        return train_labels, val_labels, test_labels

    """
    Evaluate and Tune
        - Evaluate on test Data set
    """
    def evaluate_model(self, model, val_dataloader):
        print("\nEvaluating...")
        model.to(device)
        model.eval()

        total_loss = 0
        total_preds = []

        for step, batch in enumerate(val_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            sent_id = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                preds = model(sent_id, mask)
                loss = MSELoss()(preds, labels)
                total_loss += loss.item()
                total_preds.append(preds)

        avg_loss = total_loss / len(val_dataloader)
        total_preds = torch.cat(total_preds, dim=0)

        return avg_loss, total_preds

    """
    Export Model & Inference Script
        - Save trained model weights
    """
    def export_weights(self):
        try:
            torch.save(self.to('cpu').state_dict(), self.decepticons_weights_path)
            self.to(device)  # Move back to original device
            print(f"Saved weights to {self.decepticons_weights_path}")
        except Exception as e:
            print(f"Failed to save weights: {e}")

    def load_weights(self):
        print("Loading weights...")
        print(f"Loading weights from {self.decepticons_weights_path}")
        try:
            state_dict = torch.load(self.decepticons_weights_path, map_location=device)
            self.load_state_dict(state_dict)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            # Optionally: Delete corrupted file and retrain

    """
    Train the Model
        - Define loss (e.g., MSELoss) and optimizer (e.g., Adam)
        - Set up training loop with validation
        - @source https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    """
    def optimize_model(self, model, train_dataloader):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        model.train()
        total_loss = 0
        total_preds = []

        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

            sent_id = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            preds = model(sent_id, mask)
            loss_fn = MSELoss()
            loss = loss_fn(preds, labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_preds.append(preds.detach())

        avg_loss = total_loss / len(train_dataloader)
        total_preds = torch.cat(total_preds, dim=0)

        return avg_loss, total_preds

    @staticmethod
    def test_model(model, test_dataloader):
        print("\nTesting...")
        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        for batch in test_dataloader:
            sent_id = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                preds = model(sent_id, mask)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)

        return mse, mae, r2, all_preds

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

if __name__ == "__main__":

        dec = decepticons()
        dec.load_weights()
        dec.eval()
        dec.to(device)

        # Split data
        train_df, val_df, test_df = dec.split_data()

        # Create binary labels from price
        train_labels, val_labels, test_labels = dec.discretize_labels(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df
        )

        # Tokenize
        tokens_train, tokens_val, _ = dec.generate_embeddings(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df
        )

        _, _, tokens_test = dec.generate_embeddings(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df
        )

        # Dataset & DataLoader
        train_dataset = ReviewDataset(tokens_train, train_labels.tolist())
        val_dataset = ReviewDataset(tokens_val, val_labels.tolist())
        dataset = ReviewDataset(tokens_test, test_labels.tolist())
        print(f"test labels: {test_labels}")
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)
        test_loader = DataLoader(dataset, batch_size=8, num_workers=4)

        if dec.test_mode:
            mse, mae, r2, preds =  dec.test_model(dec, test_loader)
            print(f"predsictions: {preds}")

        elif dec.prediction_mode:
            dec.predict_review_score_values(dec, test_loader)
        else:
            for epoch in range(10):
                print("Epoch {}/{}".format(epoch + 1, 10))
                # Train model
                dec.optimize_model(dec, train_loader)
                # Evaluate
                val_loss, val_preds = dec.evaluate_model(dec, val_loader)
                print("Validation Loss:", val_loss)
                dec.export_weights()

        # Use actual continuous review scores instead of discretized labels
        true_continuous = test_df["review_scores_value"].values  # continuous values

        # Get indices to use as x-axis
        indices = np.arange(len(true_continuous))

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=indices, y=true_continuous, label='Actual', alpha=0.6)
        sns.scatterplot(x=indices, y=preds, label='Predicted', alpha=0.6)

        plt.xlabel("Sample Index")
        plt.ylabel("Review Score")
        plt.title("Actual vs. Predicted Review Scores")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("actual_vs_predicted_scatter_overlay.png")
        print("Plot saved as actual_vs_predicted_scatter_overlay.png")







