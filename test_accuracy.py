import numpy as np
import random
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import gensim.downloader as api
from aigame_testing import word2vec_calculation, hash_game_state, dehash_game_state, enforce_constraints, viterbi_with_constraints

word2vec_model = api.load("word2vec-google-news-300")

fruits = ["blueberry", "strawberry", "kiwi", "apple", 
        "banana", "orange", "mango", "grape",
        "pineapple", "watermelon", "papaya", "cherry",
        "peach", "pear", "plum", "guava"]
clothing = ["shirt", "pants", "skirt", "shorts",
        "hat", "scarf", "jacket", "dress",
        "blouse", "suit", "tie", "gloves",
        "hoodie", "socks", "jeans", "pajamas"]
colors = ["red", "orange", "yellow", "green",
        "blue", "purple", "indigo", "pink",
        "white", "gray", "black", "brown",
        "beige", "maroon", "teal", "navy"]
sports = ["soccer", "football", "basketball", "tennis",
        "cricket", "volleyball", "swimming", "golf",
        "rugby", "hockey", "badminton", "wrestling",
        "gymnastics", "running", "fencing", "snowboarding"]
flowers = ["rose", "tulip", "lily", "daisy",
    "sunflower", "orchid", "marigold", "lavender",
    "chrysanthemum", "peony", "jasmine", "daffodil",
    "hibiscus", "carnation", "iris", "lotus"]
shapes = [ "circle", "square", "triangle", "rectangle",
    "oval", "pentagon", "hexagon", "octagon",
    "diamond", "trapezoid", "parallelogram", "star",
    "heart", "crescent", "cube", "cylinder"]
cuisines = ["italian", "chinese", "indian", "mexican",
    "japanese", "thai", "french", "greek",
    "spanish", "korean", "vietnamese", "turkish",
    "moroccan", "american", "lebanese", "ethiopian"]

# categories with overlapping words
animals = ["dog", "cat", "lion", "tiger",
    "bear", "wolf", "fox", "horse",
    "shark", "whale", "snake", "eagle",
    "zebra", "mouse", "rabbit", "panda"]
verbs = ["bat", "fly", "run", "jump",
        "write", "play", "watch", "climb",
        "drive", "laugh", "read", "dance",
        "eat", "drink", "talk", "think"] # overlaps with animal fly and bat

# ambiguous sorting, each could relate to human in some way
tree = ["palm", "limb", "trunk", "crown",
        "heart", "veins", "skin", "roots",
        "branch", "sap", "knot", "leaf",
        "ring", "bark", "shoot", "stem"]
human = ["head", "arm", "leg", "torso",
         "hand", "foot", "eye", "ear",
        "nose", "mouth", "heart", "lungs",
        "skin", "brain", "stomach", "bones"]


# # get a random matrix of observations
# selected_elements = random.sample(shapes, 4) + random.sample(animals, 4) + random.sample(fruits, 4) + random.sample(colors, 4)
# random.shuffle(selected_elements)
# observations = np.array(selected_elements).reshape(16,)
# # print("Randomly ordered 4x4 matrix:")
# # print(observations)
# # set states
# states = np.array(["shapes", "animals", "colors", "fruits"])

# category_state_mapping = {
#     "shapes": states[:4],
#     "animals": states[4:8],
#     "colors": states[8:12],
#     "fruits": states[12:]
# }

def numcorrect(observations, best_path, category_state_mapping):
    correct = 0
    incorrect = 0
    
    for obs, state in zip(observations, best_path):
        # Determine the correct list for the current state
        if state == "shapes":
            state_items = shapes
        elif state == "animals":
            state_items = animals
        elif state == "colors":
            state_items = colors
        elif state == "fruits":
            state_items = fruits
        elif state == "clothing":
            state_items = clothing
        elif state == "cuisines":
            state_items = cuisines
        elif state == "sports":
            state_items = sports
        elif state == "flowers":
            state_items = flowers
        else:
            state_items = []
        
        # Check if the observation belongs to the correct category
        if obs in state_items:
            correct += 1
        else:
            #print(f"Incorrect match: {obs} -> {state}")
            incorrect += 1

    #print(f"Correct: {correct}, Incorrect: {incorrect}")
    ratio = (correct/16) * 100
    return ratio

def run_aigame(observations, states, category_state_mapping):
    # Initialize emission matrix
    emission_matrix_observations_states = np.zeros((len(states), len(observations)))
    state_similarities = []
    for state in states:
        similarities = np.array([word2vec_calculation(obs, state) for obs in observations])
        state_similarities.append(similarities)
    state_similarities = np.array(state_similarities)
    for state_index in range(len(states)):
        similarities = state_similarities[state_index]
        normalized_similarities = similarities / np.sum(similarities)
        emission_matrix_observations_states[state_index] = normalized_similarities

    # Initialize the transition matrix
    transition_matrix = np.zeros((len(states), len(states)))
    for current_state in range(len(states)):
        for next_state in range(len(states)):
            transition_matrix[current_state][next_state] = 1 / len(states)

    best_path = viterbi_with_constraints(observations, states, emission_matrix_observations_states, transition_matrix)
    return best_path

def makeplot(percents, name):
        x = np.arange(len(percents)) 
        y = np.array(percents)
        slope, intercept = np.polyfit(x, y, 1)
        best_fit_line = slope * x + intercept

        plt.figure(figsize=(12, 6))
        plt.scatter(x, y, s=5, alpha=0.6, label="Individual Trial Percentages")
        plt.plot(x, best_fit_line, color="red", linewidth=2, label="Line of Best Fit")
        plt.title(name)
        plt.xlabel("Trial Number")
        plt.ylabel("Percent Correct")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

def trials_with_best_fit(num, category_to_array):
    all_categories = list(category_to_array.keys())
    
    percents = []
    for i in range(num):
        # Select 4 random categories for this trial
        random_categories = random.sample(all_categories, 4)
        categories_arrays = [category_to_array[name] for name in random_categories]

        # Sample elements and create observations
        selected_elements = random.sample(categories_arrays[0], 4) + \
                            random.sample(categories_arrays[1], 4) + \
                            random.sample(categories_arrays[2], 4) + \
                            random.sample(categories_arrays[3], 4)
        random.shuffle(selected_elements)
        observations = np.array(selected_elements).reshape(16,)
        states = np.array(random_categories)
        category_state_mapping = {random_categories[i]: categories_arrays[i] for i in range(4)}
        
        # Run the AI game
        bp = run_aigame(observations, states, category_state_mapping)
        p = numcorrect(observations, bp, category_state_mapping)
        percents.append(p)

    # Calculate average correctness
    average = np.mean(percents)
    print("Average Correctness: " + str(average))
    makeplot(percents, "Percent Correct for Each Trial")


category_simple = {
        "fruits": fruits,
        "clothing": clothing,
        "colors": colors,
        "sports": sports,
        "flowers": flowers,
        "shapes": shapes,
        "animals": animals,
        "cuisines": cuisines
    }
# Run trials with random category selection for each trial
trials_with_best_fit(10000, category_simple)

# run trials with 3 simple + 1 tree
def trials_with_tree(num, category_simple, tree):
    filtered_categories = {key: value for key, value in category_simple.items() if key not in ["verbs", "animals"]}
    
    percents = []
    for i in range(num):
        random_categories = random.sample(list(filtered_categories.keys()), 3)
        categories_arrays = [filtered_categories[name] for name in random_categories]

        selected_categories = ["tree"] + random_categories
        selected_arrays = [tree] + categories_arrays

        selected_elements = random.sample(selected_arrays[0], 4) + \
                            random.sample(selected_arrays[1], 4) + \
                            random.sample(selected_arrays[2], 4) + \
                            random.sample(selected_arrays[3], 4)
        random.shuffle(selected_elements)
        observations = np.array(selected_elements).reshape(16,)
        states = np.array(selected_categories)
        category_state_mapping = {selected_categories[i]: selected_arrays[i] for i in range(4)}
        
        bp = run_aigame(observations, states, category_state_mapping)
        p = numcorrect(observations, bp, category_state_mapping)
        percents.append(p)

    average = np.mean(percents)
    print("Average Correctness: " + str(average))
    makeplot(percents, "Percent Correct for Each Trial - Only Tree")

# run trials for 3 simple + 1 human
def trials_with_human(num, category_simple, human):
    filtered_categories = {key: value for key, value in category_simple.items() if key not in ["verbs", "animals"]}
    
    percents = []
    for i in range(num):
        random_categories = random.sample(list(filtered_categories.keys()), 3)
        categories_arrays = [filtered_categories[name] for name in random_categories]

        selected_categories = ["human"] + random_categories
        selected_arrays = [human] + categories_arrays

        selected_elements = random.sample(selected_arrays[0], 4) + \
                            random.sample(selected_arrays[1], 4) + \
                            random.sample(selected_arrays[2], 4) + \
                            random.sample(selected_arrays[3], 4)
        random.shuffle(selected_elements)
        observations = np.array(selected_elements).reshape(16,)
        states = np.array(selected_categories)
        category_state_mapping = {selected_categories[i]: selected_arrays[i] for i in range(4)}
        
        bp = run_aigame(observations, states, category_state_mapping)
        p = numcorrect(observations, bp, category_state_mapping)
        percents.append(p)

    average = np.mean(percents)
    print("Average Correctness: " + str(average))
    makeplot(percents, "Percent Correct for Each Trial - Only Human")

# run trials for tree and human + 2 simple
def trials_with_tree_and_human(num, category_simple, tree, human):
    filtered_categories = {key: value for key, value in category_simple.items() if key not in ["verbs", "animals"]}
    
    percents = []
    for i in range(num):
        random_categories = random.sample(list(filtered_categories.keys()), 2)
        categories_arrays = [filtered_categories[name] for name in random_categories]

        selected_categories = ["tree", "human"] + random_categories
        selected_arrays = [tree, human] + categories_arrays

        selected_elements = random.sample(selected_arrays[0], 4) + \
                            random.sample(selected_arrays[1], 4) + \
                            random.sample(selected_arrays[2], 4) + \
                            random.sample(selected_arrays[3], 4)
        random.shuffle(selected_elements)
        observations = np.array(selected_elements).reshape(16,)
        states = np.array(selected_categories)
        category_state_mapping = {selected_categories[i]: selected_arrays[i] for i in range(4)}
        
        bp = run_aigame(observations, states, category_state_mapping)
        p = numcorrect(observations, bp, category_state_mapping)
        percents.append(p)

    # Calculate average correctness
    average = np.mean(percents)
    print("Average Correctness: " + str(average))
    makeplot(percents, "Percent Correct for Each Trial - Human and Tree")

trials_with_tree(10000, category_simple, tree)
# trials_with_human(10000, category_simple, human)
# trials_with_tree_and_human(10000, category_simple, tree, human)



