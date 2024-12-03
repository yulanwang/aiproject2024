import re
import pygame
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

# Load Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

def get_word_embedding(word):
    try:
        return word2vec_model[word]
    except KeyError:
        print(f"The word '{word}' is not in the Word2Vec vocabulary.")
        return np.zeros(word2vec_model.vector_size)

def word2vec_calculation(word1, word2):
    embedding1 = get_word_embedding(word1).reshape(1, -1)
    embedding2 = get_word_embedding(word2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    if similarity < 0:
        return similarity**2
    else:
        return similarity 
    
def preprocess_states(states):
    cleaned_states = [re.sub(r'\d+', '', state) for state in states]
    return cleaned_states

def run_aigame(observations, states, category_state_mapping):
    # Initialize emission matrix
    cleaned_states = preprocess_states(states)

    emission_matrix = np.zeros((len(cleaned_states), len(observations)))
    for state_index in range(len(cleaned_states)):
        for obs_index in range(len(observations)):
            similarity = word2vec_calculation(observations[obs_index], cleaned_states[state_index])
            emission_matrix[state_index, obs_index] = similarity
    emission_matrix /= emission_matrix.sum(axis=0, keepdims=True)

    transition_matrix = np.full((len(cleaned_states), len(cleaned_states)), 1 / len(cleaned_states))

    best_path = viterbi_with_constraints(observations, cleaned_states, emission_matrix, transition_matrix)
    return best_path

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

# True to display the Pygame front-end, False to run without GUI
GUI = True  

# Define categories and their colors
categories = {
    'fruit': (111, 148, 96),   # GREEN
    'season': (224, 76, 76),   # RED
    'sport': (255, 255, 0),    # YELLOW
    'cuisine': (137, 207, 240) # BLUE
}

# Define words for each category (make sure all words are lowercase)
fruits = ['strawberry', 'kiwi', 'apple', 'banana']
seasons = ['spring', 'summer', 'autumn', 'winter']
sports = ['soccer', 'basketball', 'tennis', 'volleyball']
cuisines = ['indian', 'thai', 'chinese', 'japanese']

observations_list = fruits + seasons + sports + cuisines
random.shuffle(observations_list)
observations = np.array(observations_list)

states = np.array(list(categories.keys()))

category_state_mapping = {
    'fruit': fruits,
    'season': seasons,
    'sport': sports,
    'cuisine': cuisines
}

if GUI:
    pygame.init()

    best_path = []

    # Constants
    WIDTH, HEIGHT = 800, 750  
    GRID_SIZE = 4
    WORD_BOX_WIDTH = 180
    WORD_BOX_HEIGHT = 80
    PADDING = 10
    MESSAGE_HEIGHT = 50

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BORDER_COLOR = (128, 128, 128)
    HIGHLIGHT_COLOR = (255, 215, 0)
    BUTTON_COLOR = (100, 149, 237)
    BUTTON_HOVER_COLOR = (65, 105, 225)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Solving the Connections Game")
    font = pygame.font.Font(None, 36)

    word_positions = {}
    message = "Click 'Solve' to let the AI solve the game."

    # solve button
    BUTTON_WIDTH = 150
    BUTTON_HEIGHT = 50
    solve_button_rect = pygame.Rect((WIDTH - BUTTON_WIDTH) // 2, HEIGHT - MESSAGE_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT)
    solve_again_button_rect = pygame.Rect((WIDTH - BUTTON_WIDTH) // 2, HEIGHT - MESSAGE_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT)

    # Variables to control AI solving
    ai_solving = False
    step = 0 
    highlight_word = None
    move_delay = 500  
    last_move_time = pygame.time.get_ticks()

    def draw_word_box(word, x, y, color=WHITE):
        pygame.draw.rect(screen, color, (x, y, WORD_BOX_WIDTH, WORD_BOX_HEIGHT))
        pygame.draw.rect(screen, BORDER_COLOR, (x, y, WORD_BOX_WIDTH, WORD_BOX_HEIGHT), 2)
        text = font.render(word.capitalize(), True, BLACK)
        text_rect = text.get_rect(center=(x + WORD_BOX_WIDTH // 2, y + WORD_BOX_HEIGHT // 2))
        screen.blit(text, text_rect)

    def draw_category_box(category, x, y, color):
        pygame.draw.rect(screen, color, (x, y, WIDTH // 4 - PADDING, 50))
        text = font.render(category.capitalize(), True, BLACK)
        text_rect = text.get_rect(center=(x + (WIDTH // 4 - PADDING) // 2, y + 25))
        screen.blit(text, text_rect)

    def display_message(text):
        pygame.draw.rect(screen, WHITE, (0, HEIGHT - MESSAGE_HEIGHT, WIDTH, MESSAGE_HEIGHT))
        msg_text = font.render(text, True, BLACK)
        text_rect = msg_text.get_rect(center=(WIDTH // 2, HEIGHT - MESSAGE_HEIGHT // 2))
        screen.blit(msg_text, text_rect)

    def draw_solve_button():
        mouse_pos = pygame.mouse.get_pos()
        if solve_button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, BUTTON_HOVER_COLOR, solve_button_rect)
        else:
            pygame.draw.rect(screen, BUTTON_COLOR, solve_button_rect)
        text = font.render("Solve", True, WHITE)
        text_rect = text.get_rect(center=solve_button_rect.center)
        screen.blit(text, text_rect)

    def draw_solve_again_button():
        mouse_pos = pygame.mouse.get_pos()
        if solve_again_button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, BUTTON_HOVER_COLOR, solve_again_button_rect)
        else:
            pygame.draw.rect(screen, BUTTON_COLOR, solve_again_button_rect)
        text = font.render("Solve Again", True, WHITE)
        text_rect = text.get_rect(center=solve_again_button_rect.center)
        screen.blit(text, text_rect)

    def reset_game_state():
        global ai_solving, step, highlight_word, last_move_time, best_path, message
        ai_solving = False
        step = 0
        highlight_word = None
        last_move_time = pygame.time.get_ticks()
        best_path = []
        message = "Click 'Solve' to let the AI solve the game."

    running = True
    clock = pygame.time.Clock()

    try:
        while running:
            screen.fill(WHITE)

            # Draw the grid of words
            for i, word in enumerate(observations):
                row, col = divmod(i, GRID_SIZE)
                x = col * (WORD_BOX_WIDTH + PADDING) + PADDING
                y = row * (WORD_BOX_HEIGHT + PADDING) + PADDING

                if ai_solving and i < step:
                    assigned_category = best_path[i]
                    color = categories[assigned_category]
                elif not ai_solving and best_path:
                    assigned_category = best_path[i]
                    color = categories[assigned_category]
                else:
                    color = WHITE

                if word == highlight_word:
                    draw_word_box(word, x, y, HIGHLIGHT_COLOR)
                else:
                    draw_word_box(word, x, y, color)

                word_positions[word] = (x, y)

            # Draw categories
            category_y = HEIGHT - 60 - MESSAGE_HEIGHT - PADDING - BUTTON_HEIGHT - 10  
            for i, (category, color) in enumerate(categories.items()):
                draw_category_box(category, i * (WIDTH // 4), category_y, color)

            display_message(message)

            if not ai_solving and not best_path:
                draw_solve_button()
            elif not ai_solving and best_path:
                draw_solve_again_button()

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not ai_solving and not best_path and solve_button_rect.collidepoint(event.pos):
                        # Start the AI solving process here
                        best_path = run_aigame(observations, states, category_state_mapping)
                        ai_solving = True
                        message = "AI is solving the game..."
                        step = 0
                        highlight_word = None
                        last_move_time = pygame.time.get_ticks()
                    elif not ai_solving and best_path and solve_again_button_rect.collidepoint(event.pos):
                        # Reset and solve again
                        reset_game_state()
                        best_path = run_aigame(observations, states, category_state_mapping)
                        ai_solving = True
                        message = "AI is solving the game again..."
                        step = 0
                        highlight_word = None
                        last_move_time = pygame.time.get_ticks()

            # AI solving logic
            if ai_solving:
                current_time = pygame.time.get_ticks()
                if current_time - last_move_time >= move_delay and step <= len(observations):
                    if step < len(observations):
                        highlight_word = observations[step]
                        assigned_category = best_path[step]
                        message = f"Assigning '{highlight_word.capitalize()}' to '{assigned_category.capitalize()}'"
                        step += 1
                        last_move_time = current_time
                    else:
                        highlight_word = None 
                        message = "AI has finished solving the game."
                        ai_solving = False 

            pygame.display.flip()
            clock.tick(30)

    except Exception as e:
        print(f"An error occurred: {e}")

    pygame.quit()
else:
    # Run the AI game without GUI and print the results
    print("AI is solving the game...\n")
    best_path = run_aigame(observations, states, category_state_mapping)
    for word, category in zip(observations, best_path):
        print(f"'{word.capitalize()}' has been assigned to '{category.capitalize()}'")
    print("\nAI has finished solving the game.")
