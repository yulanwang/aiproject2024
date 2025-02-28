import pandas as pd

# This si to fine tune the transformer model to better understand word to category relationships 
# Load your dataset
df = pd.read_csv("nyt_connections_data.csv")

# Create training sentence pairs (word-category)
train_data = []
for _, row in df.iterrows():
    train_data.append((row["Word"], row["Category"], 1))  # Positive pairs

# Convert to DataFrame for training
train_df = pd.DataFrame(train_data, columns=["sentence1", "sentence2", "label"])
train_df.to_csv("nyt_connections_train.csv", index=False)

print("Training data prepared.")
