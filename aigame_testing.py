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

# Define observations, categories, and states
# observations = np.array(["blueberry", "kiwi", "spring", "summer", 
#                          "football", "fall", "winter", "soccer",
#                          "indian", "thai", "basketball", "japanese",
#                          "volleyball", "apple", "chinese", "banana"])

# observations = np.array(["train", "bus", "panda", "red", 
#                          "lion", "pink", "teacher", "zebra",
#                          "airplane", "dog", "doctor", "chef",
#                          "yellow", "bicycle", "blue", "engineer"])

# states = np.array(["transportation", "animal", "color", "job"])
#states = np.array(["fruit", "season", "sport", "cuisine"])
# states = np.array(["Fruit1", "Fruit2", "Fruit3", "Fruit4", 
#                    "Season1", "Season2", "Season3", "Season4",
#                    "Sport1", "Sport2", "Sport3", "Sport4",
#                    "Cuisine1", "Cuisine2", "Cuisine3", "Cuisine4"])

# THESE *************
# observations = np.array(["daffodil", "shirt", "square", "rain", 
#                          "pants", "daisy", "snow", "triangle",
#                          "wind", "rectangle", "thunder", "jacket",
#                          "shorts", "circle", "lily", "tulip"])

# states = np.array(["flower", "clothing", "shapes", "weather"])

# # Define category-state mapping
# category_state_mapping = {
#     "flower": states[:4],
#     "clothing": states[4:8],
#     "shapes": states[8:12],
#     "weather": states[12:]
# }
# UNTIL HERE ****************

def hash_game_state(game_state):
    hashed_state = int("".join(map(str, game_state)))
    return hashed_state

def dehash_game_state(hashed_state):
    game_state = [int(digit) for digit in str(hashed_state)]
    return game_state

# Create the emission matrix
# emission_matrix_observations_states = np.zeros((len(states), len(observations))) THIS *************************

# cleaned_states = preprocess_states(states)

# for state_index in range(len(states)):
#     for observation_index in range(len(observations)):
#         similarity = word2vec_calculation(observations[observation_index], states[state_index])
#         probability = prob_in_state(states[state_index], similarity)
#         emission_matrix_observations_states[state_index][observation_index] = probability

# pre-compute similarities between state for each observation
# THESE **********************************************************************************************************
# state_similarities = []
# for state in states:
#     similarities = np.array([word2vec_calculation(obs, state) for obs in observations])
#     state_similarities.append(similarities)

# state_similarities = np.array(state_similarities)


# for state_index in range(len(states)):
#     similarities = state_similarities[state_index]
#     normalized_similarities = similarities / np.sum(similarities)
#     emission_matrix_observations_states[state_index] = normalized_similarities
# UNTIL HERE **********************************************************************************************************

#print(emission_matrix_observations_states)


# #TO DO: create transition matrix
# transition_matrix = np.zeros([5000, 5000], dtype=int)

# for row in range(len(transition_matrix)):
#     for column in range(len(transition_matrix)):
#          if row < 4445 and column < 4445:
#              current_state = dehash_game_state(transition_matrix[row][column])
#              for i in range(len(current_state)):
#                  current_state[i] += 1
#                  hashed_state = hash_game_state(current_state)
#                  if hashed_state < 4445:
#                      transition_matrix[row][hashed_state] = 1
#                      current_state[i] -= 1
#                  else:
#                      transition_matrix[row][hashed_state] = 0

# print("Emission Matrix (Categories x States):")
# print(emission_matrix_observations_states)
# print("Emission Matrix (Observations x States):")
# print(emission_matrix_observations_states)

# Example similarity calculations
# similarity_fruit = word2vec_calculation('Blueberry', 'Fruit')
# similarity_season = word2vec_calculation('Blueberry', 'Season')
# similarity_sport = word2vec_calculation('Blueberry', 'Sport')
# similarity_cuisine = word2vec_calculation('Blueberry', 'Cuisine')

# print(f"Cosine similarity between 'Blueberry' and 'Fruit': {similarity_fruit}")
# print(f"Cosine similarity between 'Blueberry' and 'Season': {similarity_season}")
# print(f"Cosine similarity between 'Blueberry' and 'Sport': {similarity_sport}")
# print(f"Cosine similarity between 'Blueberry' and 'Cuisine': {similarity_cuisine}")

# represents 0 for fruits filled, 1 season filled, 0 for cuisines filled, 0 for sports filled

# THESE **********************************************************************************************************
# game_state = [0, 1, 0, 0]

# # Hash the game state
# hashed_value = hash_game_state(game_state)
# print(f"Hashed value: {hashed_value}")

# # Dehash the value back to the game state
# dehashed_state = dehash_game_state(hashed_value)
# print(f"Dehashed game state: {dehashed_state}")
# UNTIL HERE **********************************************************************************************************

#print(transition_matrix)

#DUMMY TRANSITION MATRIX DOESN"T WORK
# # Transition matrix - favor transitions within the same category
# transition_matrix = np.zeros((len(states), len(states)))
# for i, current_state in enumerate(states):
#     for j, next_state in enumerate(states):
#         # Transitions are more probable within the same category (e.g., Fruit1 -> Fruit2)
#         if current_state[:-1] == next_state[:-1]:  # Same category
#             transition_matrix[i, j] = 0.8  # Higher probability for within-category transitions
#         else:
#             transition_matrix[i, j] = 0.2 / (len(states) - 4)  # Lower probability for across-category transitions

# # # Normalize transition matrix
# transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

# THESE **********************************************************************************************************
# transition_matrix = np.zeros((len(states), len(states)))
# for current_state in range(len(states)):
#     for next_state in range(len(states)):
#         transition_matrix[current_state][next_state] = 1/4
# UNTIL HERE **********************************************************************************************************

# IMPLEMENT VIRTERBI ALGORITHM HERE:

# Calculate the stationary distribution
# transition_matrix_transp = transition_matrix.T
# eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
# '''
# Find the indexes of the eigenvalues that are close to one.
# Use them to select the target eigen vectors. Flatten the result.
# '''
# close_to_1_idx = np.isclose(eigenvals,1)
# target_eigenvect = eigenvects[:,close_to_1_idx]
# target_eigenvect = target_eigenvect[:,0]
# # Turn the eigenvector elements into probabilites
# stationary_distrib = target_eigenvect / sum(target_eigenvect) 

# '''
# Find the indexes of the eigenvalues that are close to one.
# Use them to select the target eigen vectors. Flatten the result.
# '''
# close_to_1_idx = np.isclose(eigenvals,1)
# target_eigenvect = eigenvects[:,close_to_1_idx]
# target_eigenvect = target_eigenvect[:,0]
# # Turn the eigenvector elements into probabilites
# stationary_distrib = target_eigenvect / sum(target_eigenvect) 

# def enforce_constraints(best_path, states):
#     # Map each state to its category
#     state_to_category = {}
#     for category, state_list in category_state_mapping.items():
#         for state in state_list:
#             state_to_category[state] = category

#     # Track how many times each category is assigned
#     category_counts = {category: 0 for category in category_state_mapping.keys()}
#     max_assignments = len(observations) // len(states)  # Equal division across categories

#     # Final path considering constraints
#     final_path = []
#     for state_idx in best_path:
#         state = states[state_idx]
#         category = state_to_category.get(state, None)

#         if category and category_counts[category] < max_assignments:
#             final_path.append(state_idx)
#             category_counts[category] += 1
#         else:
#             # If max assignments for this category, fallback to the next state
#             for alt_state_idx in range(len(states)):
#                 alt_state = states[alt_state_idx]
#                 alt_category = state_to_category.get(alt_state, None)
#                 if alt_category and category_counts[alt_category] < max_assignments:
#                     final_path.append(alt_state_idx)
#                     category_counts[alt_category] += 1
#                     break

#     return final_path


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

# THESE **********************************************************************************************************
# emission_matrix_observations_states = np.zeros((len(states), len(observations)))

# for state_index in range(len(states)):
#     similarities = state_similarities[state_index]
#     emission_matrix_observations_states[state_index] = similarities / np.sum(similarities)

# best_path = viterbi_with_constraints(observations, states, emission_matrix_observations_states, transition_matrix)

# print("Best path of states for given observations:")
# for obs, state in zip(observations, best_path):
#     print(f"{obs} -> {state}")

# UNTIL HERE **********************************************************************************************************

# observations = np.array(["blueberry", "kiwi", "spring", "summer", 
#                           "football", "fall", "winter", "soccer",
#                           "indian", "thai", "basketball", "japanese",
#                           "volleyball", "apple", "chinese", "banana"])
# states = np.array(["fruit", "season", "sport", "cuisine"])

# cleaned_states = preprocess_states(states)

# emission_matrix = np.zeros((len(cleaned_states), len(observations)))
# for state_index in range(len(cleaned_states)):
#     for obs_index in range(len(observations)):
#         similarity = word2vec_calculation(observations[obs_index], cleaned_states[state_index])
#         emission_matrix[state_index, obs_index] = similarity
# emission_matrix /= emission_matrix.sum(axis=0, keepdims=True)

# transition_matrix = np.full((len(cleaned_states), len(cleaned_states)), 1 / len(cleaned_states))

# best_path = viterbi_with_constraints(observations, cleaned_states, emission_matrix, transition_matrix)

# print("Best path of states for given observations:")
# for obs, state in zip(observations, best_path):
#     print(f"{obs} -> {state}")

