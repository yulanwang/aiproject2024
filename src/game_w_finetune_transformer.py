from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Load the fine-tuned model
model = SentenceTransformer("fine_tuned_nyt_connections_model")

# Load NYT Connections dataset (example)
df = pd.read_csv("nyt_connections_data.csv")


def get_category_similarities(words, categories):
    """Compute similarity scores between words and categories using fine-tuned Transformer model."""
    
    # Handle empty input case
    if not words or not categories:
        raise ValueError("Error: Words or Categories list is empty!")
    
    # Encode words and categories
    word_embeddings = model.encode(words)
    category_embeddings = model.encode(categories)

    similarity_matrix = cosine_similarity(word_embeddings, category_embeddings)
    return similarity_matrix



def viterbi_with_constraints(words, categories, similarity_matrix):
    """
    Viterbi-style approach to assign each word to exactly one category
    while ensuring balanced category assignment.
    """
    assigned_categories = {}
    
    # Sort similarity scores
    word_assignments = np.argmax(similarity_matrix, axis=1)
    
    # Ensure each category gets 4 words
    category_counts = {category: 0 for category in categories}
    final_assignments = {}
    
    for i, word in enumerate(words):
        best_category = categories[word_assignments[i]]
        
        # If the category already has 4 words, pick the next best category
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        
        for idx in sorted_indices:
            category_choice = categories[idx]
            if category_counts[category_choice] < 4:
                final_assignments[word] = category_choice
                category_counts[category_choice] += 1
                break
    
    return final_assignments


def play_game(game_id):
    """Simulates AI playing a round of NYT Connections."""
    
    # Extract game data
    game_data = df[df["Game ID"] == game_id]

    # Debugging: Print dataset structure
    print(f"\n🔹 Checking data for Game ID {game_id}")
    print(game_data)  # Print the full game data to see what's inside

    # Check if the dataframe is actually empty
    if game_data.empty:
        print(f"⚠️ Game ID {game_id} does not exist in the dataset!")
        return None

    # Extract words and categories
    words = game_data["Word"].dropna().tolist()
    categories = game_data["Category"].dropna().unique().tolist()

    # Debugging: Print extracted words & categories
    print(f"\n🔹 Extracted Words: {words}")
    print(f"🔹 Extracted Categories: {categories}")

    # If either is empty, skip the game
    if len(words) == 0 or len(categories) == 0:
        print(f"⚠️ Skipping Game {game_id}: No words or categories found!")
        return None

    # Compute similarity matrix
    similarity_matrix = get_category_similarities(words, categories)

    # Assign words to categories
    assigned_categories = viterbi_with_constraints(words, categories, similarity_matrix)

    # Display results
    print(f"\n🔹 AI Results for Game ID {game_id} 🔹")
    for word, category in assigned_categories.items():
        print(f"{word} → {category}")

    return assigned_categories


def evaluate_performance():
    """Run AI on all games and display accuracy results."""
    game_ids = df["Game ID"].unique()
    results = []

    for game_id in game_ids:
        # Extract words and categories for the game
        game_data = df[df["Game ID"] == game_id]
        words = game_data["Word"].tolist()
        categories = game_data["Category"].unique().tolist()

        # Compute similarity matrix
        similarity_matrix = get_category_similarities(words, categories)

        # Get AI predictions
        predicted_labels = viterbi_with_constraints(words, categories, similarity_matrix)

        # Extract actual labels
        true_labels = game_data.set_index("Word")["Category"].to_dict()

        # Compute accuracy
        y_true = list(true_labels.values())
        y_pred = [predicted_labels[word] for word in true_labels.keys()]
        accuracy = accuracy_score(y_true, y_pred)

        results.append({"Game ID": game_id, "Accuracy": accuracy})

        print(f"Game {game_id}: AI Accuracy = {accuracy:.2%}")

    return results

results = evaluate_performance()

# Convert results to DataFrame
results_df = pd.DataFrame(results)
# Save results to CSV
results_df.to_csv("ai_performance_results.csv", index=False)
print("\n✅ Results saved to 'ai_performance_results.csv'")

custom_color = (179/255, 167/255, 254/255) 
# Plot accuracy for each game and save it
plt.figure(figsize=(12, 5))
plt.bar(results_df["Game ID"].astype(str), results_df["Accuracy"], color=custom_color)
plt.xlabel("Game ID")
plt.ylabel("Accuracy")
plt.title("Transformer Model Performance Across Games")
plt.xticks(rotation=90)  # Rotate game IDs for readability
plt.ylim(0, 1)

# Save the figure before showing it
plt.savefig("ai_performance_chart.png", dpi=300, bbox_inches="tight")
print("\n📊 Chart saved as 'ai_performance_chart.png'")

plt.show()
