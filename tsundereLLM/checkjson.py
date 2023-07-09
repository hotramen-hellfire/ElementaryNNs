import json

# Assuming your JSON file is named "intents.json"
file_path = "./LLMS/intents.json"

# Load the JSON file
with open(file_path, "r") as file:
    data = json.load(file)

# Print the loaded data
print(data)
