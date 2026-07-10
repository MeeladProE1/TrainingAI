import os
import csv
import json
import tkinter as tk
from tkinter import messagebox
from sentence_transformers import SentenceTransformer

# 1. Set up the target folder right where the script runs
FOLDER_NAME = "VectorDB"
INDEX_FILE_PATH = os.path.join(FOLDER_NAME, "index.json")

if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)

# 2. Load the lightweight local AI model
print("Loading local AI model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("AI Model loaded successfully. Launching GUI...")

def get_next_doc_name():
    """Automatically calculates the next document name sequentially (doc_001, doc_002, etc.)."""
    if not os.path.exists(FOLDER_NAME):
        return "doc_001"
        
    existing_files = [f for f in os.listdir(FOLDER_NAME) if f.startswith("doc_") and f.endswith(".csv")]
    if not existing_files:
        return "doc_001"
    
    # Extract numbers from file names and find the maximum number
    numbers = []
    for f in existing_files:
        try:
            # Splits 'doc_001.csv' into ['doc', '001'] and parses the number
            parts = f.replace(".csv", "").split("_")
            if len(parts) > 1:
                numbers.append(int(parts[1]))
        except (IndexError, ValueError):
            continue
            
    next_num = max(numbers) + 1 if numbers else 1
    return f"doc_{next_num:03d}"

def save_data():
    """Processes the text inputs, generates the vector, and saves files securely."""
    category = entry_category.get().strip()
    data_content = text_data.get("1.0", tk.END).strip()
    
    # Validation checks
    if not category:
        messagebox.showerror("Error", "Category field cannot be empty.")
        return
    if not data_content:
        messagebox.showerror("Error", "Data field cannot be empty.")
        return

    # A. Determine the automatic sequential ID
    doc_id = get_next_doc_name()
    csv_file_name = f"{doc_id}.csv"
    csv_file_path = os.path.join(FOLDER_NAME, csv_file_name)

    try:
        # B. Generate the mathematical vector locally
        raw_vector = model.encode(data_content)
        vector_list = raw_vector.tolist()

        # C. Create / Update the main index.json file (IDs and Vectors only)
        index_data = []
        if os.path.exists(INDEX_FILE_PATH):
            with open(INDEX_FILE_PATH, "r", encoding="utf-8") as f:
                try:
                    index_data = json.load(f)
                except json.JSONDecodeError:
                    index_data = []
                    
        index_data.append({"id": doc_id, "vector": vector_list})
        
        with open(INDEX_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=4)

        # D. Save the unique categorical document file (ID, Category, Data)
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "category", "data"])
            writer.writerow([doc_id, category, data_content])

        # E. Update the UI with success information
        messagebox.showinfo("Success", f"Saved successfully!\nGenerated: {csv_file_name}")
        
        # Clear fields and reset the name label for the next entry
        entry_category.delete(0, tk.END)
        text_data.delete("1.0", tk.END)
        label_doc_name.config(text=f"Next Document Name: {get_next_doc_name()}")

    except Exception as e:
        messagebox.showerror("System Error", f"An error occurred: {str(e)}")

# --- 3. Build the Tkinter Graphical Window ---
root = tk.Tk()
root.title("AI Vector Data Creator")
root.geometry("450x420")
root.resizable(False, False)

# Visual Header / Document Name Tracker
label_doc_name = tk.Label(root, text=f"Next Document Name: {get_next_doc_name()}", font=("Arial", 11, "bold"), fg="#2c3e50")
label_doc_name.pack(pady=15)

# Category Input Section
frame_category = tk.Frame(root)
frame_category.pack(fill="x", padx=20, pady=5)
label_category = tk.Label(frame_category, text="Category (e.g., support, school, notes):", font=("Arial", 10))
label_category.pack(anchor="w")
entry_category = tk.Entry(frame_category, font=("Arial", 10), bg="#fcfcfc")
entry_category.pack(fill="x", pady=5)

# Data Input Section
frame_data = tk.Frame(root)
frame_data.pack(fill="both", expand=True, padx=20, pady=10)
label_data = tk.Label(frame_data, text="Enter Data Content:", font=("Arial", 10))
label_data.pack(anchor="w")
text_data = tk.Text(frame_data, font=("Arial", 10), bg="#fcfcfc", height=8)
text_data.pack(fill="both", expand=True, pady=5)

# Process Trigger Button
btn_save = tk.Button(root, text="Generate Vector & Save", font=("Arial", 11, "bold"), bg="#2ecc71", fg="white", command=save_data, cursor="hand2")
btn_save.pack(fill="x", padx=20, pady=20)

root.mainloop()
