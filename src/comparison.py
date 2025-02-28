import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

import pandas as pd
import numpy as np
import gensim.downloader as api
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer  # Needed for fine-tuned model
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("nyt_connections_data.csv")
games = df.groupby("Game ID")
unique_categories = df["Category"].unique().tolist()

# Load Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Load Pretrained Transformer Model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
transformer_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load Fine-Tuned Transformer Model
fine_tuned_model = SentenceTransformer("fine_tuned_nyt_connections_model")


# Functions to retrieve word embeddings
def get_word_embedding(word, model_type="word2vec"):
    """Retrieve word embedding based on the model type."""
    if model_type == "word2vec":
        try:
            return word2vec_model[word]
        except KeyError:
            return np.zeros(word2vec_model.vector_size)

    elif model_type == "transformer":
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = transformer_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    elif model_type == "fine_tuned":
        return fine_tuned_model.encode(word, convert_to_numpy=True)


# Compute similarity
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


# Generate emission matrices
def generate_emission_matrix(observations, states, model_type="word2vec"):
    """Generate emission matrix using Word2Vec, Pretrained Transformer, or Fine-Tuned Transformer."""
    emission_matrix = np.zeros((len(states), len(observations)))

    if model_type == "fine_tuned":
        obs_embeddings = fine_tuned_model.encode(observations, convert_to_numpy=True)
        state_embeddings = fine_tuned_model.encode(states, convert_to_numpy=True)
    else:
        obs_embeddings = np.array([get_word_embedding(obs, model_type) for obs in observations])
        state_embeddings = np.array([get_word_embedding(state, model_type) for state in states])

    similarity_matrix = cosine_similarity(state_embeddings, obs_embeddings)
    return similarity_matrix / (similarity_matrix.sum(axis=0, keepdims=True) + 1e-10)



def viterbi_with_constraints(words, categories, similarity_matrix, transition_matrix):
    """
    Viterbi-style approach to assign each word to exactly one category
    while ensuring balanced category assignment and considering transition probabilities.
    """
    n_observations = len(words)
    n_states = len(categories)
    
    # Initialize Viterbi DP table and backpointer
    dp = np.zeros((n_states, n_observations))
    backpointer = np.zeros((n_states, n_observations), dtype=int)

    # Initialize first column of Viterbi table
    for s in range(n_states):
        dp[s, 0] = (1 / n_states) * similarity_matrix[s, 0]  # Initial probability
        backpointer[s, 0] = -1

    # Viterbi Algorithm with Transition Probabilities
    for t in range(1, n_observations):
        for s in range(n_states):
            probabilities = [
                dp[prev_s, t - 1] * transition_matrix[prev_s, s] * similarity_matrix[s, t]
                for prev_s in range(n_states)
            ]
            dp[s, t] = max(probabilities)
            backpointer[s, t] = np.argmax(probabilities)

    # Backtrace to find best path
    best_path = np.zeros(n_observations, dtype=int)
    best_path[-1] = np.argmax(dp[:, -1])  # Start from the best final state

    for t in range(n_observations - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]

    # Convert indices back to category names
    return [categories[state_idx] for state_idx in best_path]




# Run AI game
def run_aigame(observations, states, model_type="word2vec"):
    """Runs the AI game using the specified model type."""
    emission_matrix = generate_emission_matrix(observations, states, model_type)
    transition_matrix = np.full((len(states), len(states)), 1 / len(states))  # Initialize uniform transition probabilities
    best_path = viterbi_with_constraints(observations, states, emission_matrix, transition_matrix)  # ✅ Works Now
    return best_path

# Evaluate models
def evaluate_models():
    results = []

    for game_id, game_data in games:
        words = game_data["Word"].tolist()
        actual_categories = dict(zip(game_data["Word"], game_data["Category"]))

        # Run all three models
        predictions_word2vec = run_aigame(words, unique_categories, model_type="word2vec")
        predictions_transformer = run_aigame(words, unique_categories, model_type="transformer")
        predictions_fine_tuned = run_aigame(words, unique_categories, model_type="fine_tuned")

        correct_word2vec = sum(
            [1 for word in words if predictions_word2vec[words.index(word)] == actual_categories[word]]
        )
        correct_transformer = sum(
            [1 for word in words if predictions_transformer[words.index(word)] == actual_categories[word]]
        )
        correct_fine_tuned = sum(
            [1 for word in words if predictions_fine_tuned[words.index(word)] == actual_categories[word]]
        )

        total_words = len(words)

        accuracy_word2vec = correct_word2vec / total_words
        accuracy_transformer = correct_transformer / total_words
        accuracy_fine_tuned = correct_fine_tuned / total_words

        results.append(
            {
                "Game ID": game_id,
                "Accuracy Word2Vec": accuracy_word2vec,
                "Accuracy Transformer": accuracy_transformer,
                "Accuracy Fine-Tuned Transformer": accuracy_fine_tuned,
            }
        )

        print(
            f"Game {game_id}: Word2Vec = {accuracy_word2vec:.2%} | Transformer = {accuracy_transformer:.2%} | Fine-Tuned = {accuracy_fine_tuned:.2%}"
        )

    return pd.DataFrame(results)


# Run evaluation
results_df = evaluate_models()

# Save results
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n✅ Results saved to 'model_comparison_results.csv'")

# Plot results side-by-side
fig, ax = plt.subplots(figsize=(12, 5))

bar_width = 0.3
x_indexes = np.arange(len(results_df))

ax.bar(
    x_indexes - bar_width,
    results_df["Accuracy Word2Vec"],
    bar_width,
    label="Word2Vec",
    color="blue",
)
ax.bar(
    x_indexes,
    results_df["Accuracy Transformer"],
    bar_width,
    label="Pretrained Transformer",
    color="red",
)
ax.bar(
    x_indexes + bar_width,
    results_df["Accuracy Fine-Tuned Transformer"],
    bar_width,
    label="Fine-Tuned Transformer",
    color="green",
)

ax.set_xlabel("Game ID")
ax.set_ylabel("Accuracy")
ax.set_title("Comparison of Word2Vec vs Transformer vs Fine-Tuned Transformer")
ax.set_xticks(x_indexes)
ax.set_xticklabels(results_df["Game ID"].astype(str), rotation=90)
ax.set_ylim(0, 1)
ax.legend()

# Save and show plot
plt.savefig("full_model_comparison_chart.png", dpi=300, bbox_inches="tight")
print("\n📊 Chart saved as 'full_model_comparison_chart.png'")

plt.show()
