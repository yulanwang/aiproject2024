import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


word2vec_model = KeyedVectors.load_word2vec_format("/Users/yash/Downloads/GoogleNews-vectors-negative300.bin", binary=True) 
observations = np.array(["Strawberry", "Kiwi", "Spring", "Summer", 
                         "Tennis", "Fall", "Winter", "Soccer",
                         "Indian", "Thai", "Basketball", "Japanese",
                         "Fall", "Apple", "Chinese", "Banana"])

states = np.array(["Fruit1", "Fruit2", "Fruit3", "Fruit4", 
                   "Season1", "Season2", "Season3", "Season4",
                   "Sport1", "Sport2", "Sport3", "Sport4",
                   "Cuisine1", "Cuisine2", "Cuisine3", "Cuisine4"])

state_categories = {
    "Fruits": ["Fruit1", "Fruit2", "Fruit3", "Fruit4"],
    "Seasons": ["Season1", "Season2", "Season3", "Season4"],
    "Sports": ["Sport1", "Sport2", "Sport3", "Sport4"],
    "Cuisines": ["Cuisine1", "Cuisine2", "Cuisine3", "Cuisine4"]
}

def word2vec_calculation(word, state):
    for category, states in state_categories.items():
        if state in states:
            state = category
            break
    if word in word2vec_model and state in word2vec_model:
        return cosine_similarity([word2vec_model[word]], [word2vec_model[state]])[0, 0]
    return 0

def build_emission_matrix(observations, states):
    n_observations = len(observations)
    n_states = len(states)
    emission_matrix = np.zeros((n_observations, n_states))
    for obs_idx, obs in enumerate(observations):
        for state_idx, state in enumerate(states):
            emission_matrix[obs_idx, state_idx] = word2vec_calculation(obs, state)
    emission_matrix = emission_matrix / emission_matrix.sum(axis=1, keepdims=True)
    return emission_matrix

n_states = len(states)
transition_matrix = np.full((n_states, n_states), 1.0 / n_states)
start_prob = np.full(n_states, 1.0 / n_states)
emission_matrix = build_emission_matrix(observations, states)

print(transition_matrix)
print(emission_matrix)

def enforce_constraints(best_path, states):
    category_assignments = {category: 0 for category in state_categories.keys()}
    final_path = []

    for state_idx in best_path:
        state = states[state_idx]
        for category, state_list in state_categories.items():
            if state in state_list and category_assignments[category] < 4:
                final_path.append(state_idx)
                category_assignments[category] += 1
                break

    return final_path

def viterbi_decode_with_constraints(observations, states, start_prob, transition_matrix, emission_matrix):
    n_observations = len(observations)
    n_states = len(states)
    viterbi_table = np.zeros((n_states, n_observations))
    backpointer = np.zeros((n_states, n_observations), dtype=int)

    for s in range(n_states):
        viterbi_table[s, 0] = start_prob[s] * emission_matrix[0, s]

    for t in range(1, n_observations):
        for s in range(n_states):
            probabilities = [
                viterbi_table[s_prev, t - 1] * transition_matrix[s_prev, s] * emission_matrix[t, s]
                for s_prev in range(n_states)
            ]
            viterbi_table[s, t] = max(probabilities)
            backpointer[s, t] = np.argmax(probabilities)

    best_path = np.zeros(n_observations, dtype=int)
    best_path[-1] = np.argmax(viterbi_table[:, -1])
    for t in range(n_observations - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]

    best_path = enforce_constraints(best_path, states)
    return best_path

hidden_states = viterbi_decode_with_constraints(observations, states, start_prob, transition_matrix, emission_matrix)

for obs, state_idx in zip(observations, hidden_states):
    print(f"Observation: {obs}, Hidden State: {states[state_idx]}")
