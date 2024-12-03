import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim.downloader as api

word2vec_model = api.load("word2vec-google-news-300")

def get_word_embedding(word):
    try:
        return word2vec_model[word]
    except KeyError:
        print(f"The word '{word}' is not in the Word2Vec vocabulary.")
        return np.zeros(word2vec_model.vector_size)

# Calculate cosine similarity
def word2vec_calculation(word1, word2):
    embedding1 = get_word_embedding(word1).reshape(1, -1)
    embedding2 = get_word_embedding(word2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    if similarity < 0:
        return similarity**2 # Ensuring similarity is non-negative
    else:
        return similarity 


# Preprocess state names by removing numeric suffixes
def preprocess_states(states):
    cleaned_states = [re.sub(r'\d+', '', state) for state in states]
    return cleaned_states

# Calculate the probability of a word belonging to a state
def prob_in_state(observation, theta_of_similarity):
    thetas_for_categories = []
    for index in range(len(categories)):
        co_similarity = word2vec_calculation(observation, categories[index])
        thetas_for_categories.append(co_similarity)
    probability = theta_of_similarity / np.sum(thetas_for_categories)
    return probability


def hash_game_state(game_state):
    hashed_state = int("".join(map(str, game_state)))
    return hashed_state

def dehash_game_state(hashed_state):
    game_state = [int(digit) for digit in str(hashed_state)]
    return game_state


def enforce_constraints(best_path, states):
    state_to_category = {}
    for category, state_list in category_state_mapping.items():
        for state in state_list:
            state_to_category[state] = category
    category_counts = {category: 0 for category in category_state_mapping.keys()}
    max_assignments = len(observations) // len(category_state_mapping)  

    # Final path considering constraints
    final_path = []
    for state_idx in best_path:
        state = states[state_idx]
        category = state_to_category.get(state, None)

        if category and category_counts[category] < max_assignments:
            final_path.append(state_idx)
            category_counts[category] += 1
        else:
            for alt_state_idx in range(len(states)):
                alt_state = states[alt_state_idx]
                alt_category = state_to_category.get(alt_state, None)
                if alt_category and category_counts[alt_category] < max_assignments:
                    final_path.append(alt_state_idx)
                    category_counts[alt_category] += 1
                    break

    return final_path


def viterbi_with_constraints(observations, states, emission_matrix, transition_matrix):
    n_observations = len(observations)
    n_states = len(states)

    dp = np.zeros((n_states, n_observations))
    backpointer = np.zeros((n_states, n_observations), dtype=int)

    # Initialize DP table
    for s in range(n_states):
        dp[s, 0] = (1 / n_states) * emission_matrix[s, 0]
        backpointer[s, 0] = -1

    # Viterbi algorithm
    for t in range(1, n_observations):
        for s in range(n_states):
            probabilities = [
                dp[prev_s, t - 1] * transition_matrix[prev_s, s] * emission_matrix[s, t]
                for prev_s in range(n_states)
            ]
            dp[s, t] = max(probabilities)
            backpointer[s, t] = np.argmax(probabilities)

    # Backtrace for best path
    best_path = np.zeros(n_observations, dtype=int)
    best_path[-1] = np.argmax(dp[:, -1])

    for t in range(n_observations - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]
    best_state_sequence = [states[state_idx] for state_idx in best_path]

    return best_state_sequence

