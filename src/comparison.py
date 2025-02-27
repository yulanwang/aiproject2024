import pandas as pd
import numpy as np
import gensim.downloader as api
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("nyt_connections_data.csv")
games = df.groupby("Game ID")
unique_categories = df["Category"].unique().tolist()

# Load Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Load Transformer model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
transformer_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Functions to retrieve word embeddings
def get_word2vec_embedding(word):
    try:
        return word2vec_model[word]
    except KeyError:
        return np.zeros(word2vec_model.vector_size)

def get_transformer_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Compute similarity
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# Generate emission matrices
def generate_emission_matrix(observations, states, model_type="word2vec"):
    emission_matrix = np.zeros((len(states), len(observations)))
    for state_index, state in enumerate(states):
        for obs_index, obs in enumerate(observations):
            if model_type == "word2vec":
                similarity = calculate_similarity(get_word2vec_embedding(obs), get_word2vec_embedding(state))
            else:
                similarity = calculate_similarity(get_transformer_embedding(obs), get_transformer_embedding(state))
            emission_matrix[state_index, obs_index] = similarity
    emission_matrix /= emission_matrix.sum(axis=0, keepdims=True) + 1e-10
    return emission_matrix

# Viterbi algorithm
def viterbi_with_constraints(observations, states, emission_matrix, transition_matrix):
    n_observations, n_states = len(observations), len(states)
    dp = np.zeros((n_states, n_observations))
    backpointer = np.zeros((n_states, n_observations), dtype=int)
    
    for s in range(n_states):
        dp[s, 0] = (1 / n_states) * emission_matrix[s, 0]
        backpointer[s, 0] = -1
    
    for t in range(1, n_observations):
        for s in range(n_states):
            probabilities = [
                dp[prev_s, t - 1] * transition_matrix[prev_s, s] * emission_matrix[s, t]
                for prev_s in range(n_states)
            ]
            dp[s, t] = max(probabilities)
            backpointer[s, t] = np.argmax(probabilities)
    
    best_path = np.zeros(n_observations, dtype=int)
    best_path[-1] = np.argmax(dp[:, -1])
    for t in range(n_observations - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]
    
    return [states[state_idx] for state_idx in best_path]

# Run AI game
def run_aigame(observations, states, model_type="word2vec"):
    emission_matrix = generate_emission_matrix(observations, states, model_type)
    transition_matrix = np.full((len(states), len(states)), 1 / len(states))
    best_path = viterbi_with_constraints(observations, states, emission_matrix, transition_matrix)
    return best_path

# Evaluate models
def evaluate_models():
    total_words = 0
    correct_predictions = {"word2vec": 0, "transformer": 0}
    
    for game_id, game_data in games:
        words = game_data["Word"].tolist()
        actual_categories = dict(zip(game_data["Word"], game_data["Category"]))
        
        predictions_word2vec = run_aigame(words, unique_categories, model_type="word2vec")
        predictions_transformer = run_aigame(words, unique_categories, model_type="transformer")
        
        for word in words:
            if predictions_word2vec[words.index(word)] == actual_categories[word]:
                correct_predictions["word2vec"] += 1
            if predictions_transformer[words.index(word)] == actual_categories[word]:
                correct_predictions["transformer"] += 1
            total_words += 1
    
    accuracy_word2vec = correct_predictions["word2vec"] / total_words
    accuracy_transformer = correct_predictions["transformer"] / total_words
    
    print(f"Accuracy of Word2Vec Model: {accuracy_word2vec:.2%}")
    print(f"Accuracy of Transformer Model: {accuracy_transformer:.2%}")
    
    return accuracy_word2vec, accuracy_transformer

# evaluate and compare models by graph
accuracy_word2vec, accuracy_transformer = evaluate_models()

labels = ["Word2Vec", "Transformer"]
accuracy_scores = [accuracy_word2vec, accuracy_transformer]
plt.bar(labels, accuracy_scores, color=['blue', 'red'])
plt.ylabel("Accuracy")
plt.title("Word2Vec vs Transformer Accuracy on NYT Connections")
plt.ylim(0, 1)
plt.show()