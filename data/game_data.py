import pandas as pd

# Define dataset
data = [
    [1, "sonnet", "poetic forms"],
    [1, "haiku", "poetic forms"],
    [1, "epic", "poetic forms"],
    [1, "limerick", "poetic forms"],
    [1, "solar", "renewable energy sources"],
    [1, "wind", "renewable energy sources"],
    [1, "hydro", "renewable energy sources"],
    [1, "geothermal", "renewable energy sources"],
    [1, "petra", "ancient cities"],
    [1, "machu picchu", "ancient cities"],
    [1, "pompeii", "ancient cities"],
    [1, "troy", "ancient cities"],
    [1, "hurdy-gurdy", "obscure musical instruments"],
    [1, "theremin", "obscure musical instruments"],
    [1, "zither", "obscure musical instruments"],
    [1, "sheng", "obscure musical instruments"],

    [2, "iliad", "epic poems"],
    [2, "odyssey", "epic poems"],
    [2, "beowulf", "epic poems"],
    [2, "aeneid", "epic poems"],
    [2, "impressionism", "art movements"],
    [2, "cubism", "art movements"],
    [2, "surrealism", "art movements"],
    [2, "baroque", "art movements"],
    [2, "morpheme", "linguistic concepts"],
    [2, "syntax", "linguistic concepts"],
    [2, "phoneme", "linguistic concepts"],
    [2, "semantics", "linguistic concepts"],
    [2, "tesla", "inventors and scientists"],
    [2, "einstein", "inventors and scientists"],
    [2, "curie", "inventors and scientists"],
    [2, "newton", "inventors and scientists"],

    [3, "tarantino", "famous directors"],
    [3, "spielberg", "famous directors"],
    [3, "hitchcock", "famous directors"],
    [3, "kubrick", "famous directors"],
    [3, "amazon", "river systems"],
    [3, "nile", "river systems"],
    [3, "yangtze", "river systems"],
    [3, "mississippi", "river systems"],
    [3, "mozart", "classical composers"],
    [3, "bach", "classical composers"],
    [3, "beethoven", "classical composers"],
    [3, "vivaldi", "classical composers"],
    [3, "internet", "technological innovations"],
    [3, "smartphone", "technological innovations"],
    [3, "robotics", "technological innovations"],
    [3, "ai", "technological innovations"],
]

# Convert all words and categories to lowercase
data = [[game_id, word.lower(), category.lower()] for game_id, word, category in data]

# Create DataFrame
df = pd.DataFrame(data, columns=["Game ID", "Word", "Category"])

# Save to CSV
df.to_csv("nyt_connections_data.csv", index=False)

print("CSV file saved successfully with lowercase words!")
