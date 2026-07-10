import os
import csv
import json
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sentence_transformers import SentenceTransformer

# 1. Set up path constants
FOLDER_NAME = "VectorDB"
INDEX_FILE_PATH = os.path.join(FOLDER_NAME, "index.json")

# 2. Load the lightweight local AI model
print("Loading local AI model for search...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("AI Model loaded successfully. Launching Search GUI...")

def calculate_cosine_similarity(vec1, vec2):
    """Calculates how close two vectors are mathematically."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # Handle edge case division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def perform_search():
    """Takes user prompt, checks index.json, and pulls final details from the main CSV."""
    user_prompt = entry_prompt.get().strip()
    
    if not user_prompt:
        messagebox.showerror("Error", "Please enter a search prompt.")
        return

    # Clear previous results from the text box
    text_results.config(state="normal")
    text_results.delete("1.0", tk.END)

    # Check if index exists
    if not os.path.exists(INDEX_FILE_PATH):
        messagebox.showwarning("Notice", "No index file found. Please create some data first.")
        text_results.config(state="disabled")
        return

    try:
        # A. Encode user prompt into numbers
        prompt_vector = model.encode(user_prompt).tolist()

        # B. Load the mathematical index file
        with open(INDEX_FILE_PATH, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        if not index_data:
            text_results.insert(tk.END, "The index file is empty.")
            text_results.config(state="disabled")
            return

        best_id = None
        best_score = -1.0

        # C. Loop through the index to find the highest similarity score
        for item in index_data:
            similarity = calculate_cosine_similarity(prompt_vector, item["vector"])
            if similarity > best_score:
                best_score = similarity
                best_id = item["id"]

        # D. Use the winning ID to extract details from the main data CSV file
        if best_id:
            csv_file_name = f"{best_id}.csv"
            csv_file_path = os.path.join(FOLDER_NAME, csv_file_name)

            if os.path.exists(csv_file_path):
                with open(csv_file_path, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    headers = next(reader) # Skip headers
                    row = next(reader)     # Get the data row
                    
                    # Layout indexes: id=0, vector=1, category=2, data=3
                    category_data = row[2]
                    content_data = row[3]

                # E. Print results clearly to the GUI display box
                text_results.insert(tk.END, f"--- BEST MATCH FOUND ---\n\n")
                text_results.insert(tk.END, f"Document ID: {best_id}\n")
                text_results.insert(tk.END, f"Category: {category_data}\n")
                text_results.insert(tk.END, f"Similarity Score: {best_score:.4f}\n")
                text_results.insert(tk.END, f"------------------------\n\n")
                text_results.insert(tk.END, f"Data Content:\n{content_data}")
            else:
                text_results.insert(tk.END, f"Error: Index pointed to '{csv_file_name}', but the data file is missing.")
        else:
            text_results.insert(tk.END, "No match could be determined.")

    except Exception as e:
        messagebox.showerror("System Error", f"An error occurred during search: {str(e)}")
    
    text_results.config(state="disabled")

# --- 3. Build the Tkinter Search Window ---
root = tk.Tk()
root.title("AI Vector Data Searcher")
root.geometry("480x450")
root.resizable(False, False)

# Prompt Entry Section
frame_prompt = tk.Frame(root)
frame_prompt.pack(fill="x", padx=20, pady=15)
label_prompt = tk.Label(frame_prompt, text="Enter search prompt or question:", font=("Arial", 10, "bold"))
label_prompt.pack(anchor="w")
entry_prompt = tk.Entry(frame_prompt, font=("Arial", 11), bg="#fcfcfc")
entry_prompt.pack(fill="x", pady=5)

# Search Execution Button
btn_search = tk.Button(root, text="Search Vector Database", font=("Arial", 11, "bold"), bg="#3498db", fg="white", command=perform_search, cursor="hand2")
btn_search.pack(fill="x", padx=20, pady=5)

# Results Display Section
frame_results = tk.Frame(root)
frame_results.pack(fill="both", expand=True, padx=20, pady=15)
label_results = tk.Label(frame_results, text="Results:", font=("Arial", 10))
label_results.pack(anchor="w")

text_results = tk.Text(frame_results, font=("Courier", 10), bg="#f8f9fa", state="disabled", wrap="word")
text_results.pack(fill="both", expand=True, pady=5)

root.mainloop()
