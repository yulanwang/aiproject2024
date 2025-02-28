from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# Load pre-trained model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load training data
train_examples = []
df_train = pd.read_csv("nyt_connections_train.csv")

for _, row in df_train.iterrows():
    train_examples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=float(row["label"])))

# Convert into DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# Save the fine-tuned model
model.save("fine_tuned_nyt_connections_model")
print("Fine-tuned Transformer model saved.")
