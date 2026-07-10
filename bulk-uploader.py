import os
import csv
import json
from sentence_transformers import SentenceTransformer

# 1. Path configurations
FOLDER_NAME = "VectorDB"
INDEX_FILE_PATH = os.path.join(FOLDER_NAME, "index.json")
DATA_SOURCE_FILE = "bulk.json"

if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)

# 2. Check if the bulk data file exists
if not os.path.exists(DATA_SOURCE_FILE):
    print(f"ERROR: Cannot find '{DATA_SOURCE_FILE}'! Please create it first.")
    exit()

# 3. Load the local AI model
print("Loading local AI model for bulk conversion...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_next_doc_number():
    """Finds the maximum document number currently sitting in VectorDB to avoid overwriting files."""
    existing_files = [f for f in os.listdir(FOLDER_NAME) if f.startswith("doc_") and f.endswith(".csv")]
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        try:
            parts = f.replace(".csv", "").split("_")
            if len(parts) > 1:
                numbers.append(int(parts[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers) + 1 if numbers else 1

# 4. Load the JSON recipes data
with open(DATA_SOURCE_FILE, "r", encoding="utf-8") as f:
    bulk_recipes = json.load(f)

print(f"Found {len(bulk_recipes)} recipes to process. Starting vector generation...")

# Load existing index records if the file exists
index_data = []
if os.path.exists(INDEX_FILE_PATH):
    with open(INDEX_FILE_PATH, "r", encoding="utf-8") as f:
        try:
            index_data = json.load(f)
        except json.JSONDecodeError:
            index_data = []

current_doc_num = get_next_doc_number()

# 5. Process each entry automatically
for recipe in bulk_recipes:
    category = recipe["category"]
    data_content = recipe["data"]
    
    # Format the sequential ID
    doc_id = f"doc_{current_doc_num:03d}"
    csv_file_name = f"{doc_id}.csv"
    csv_file_path = os.path.join(FOLDER_NAME, csv_file_name)
    
    print(f" -> Vectorizing and writing: {csv_file_name} [{category}]")
    
    # Calculate the vector coordinates locally
    raw_vector = model.encode(data_content)
    vector_list = raw_vector.tolist()
    vector_string = json.dumps(vector_list) # Compact string string representation for the CSV layout
    
    # Append the vector data structure directly to the master index list
    index_data.append({"id": doc_id, "vector": vector_list})
    
    # Create the complete distinct CSV file structure containing the vector
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "vector", "category", "data"])
        writer.writerow([doc_id, vector_string, category, data_content])
        
    current_doc_num += 1

# 6. Save the master index tracking update
with open(INDEX_FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(index_data, f, indent=4)

print("\n*** BULK UPLOAD COMPLETE! ***")
print(f"Master tracking file updated: {INDEX_FILE_PATH}")
print(f"Individual matching document CSV data files saved into '{FOLDER_NAME}/'")
