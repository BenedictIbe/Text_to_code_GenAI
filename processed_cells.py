import re

def preprocess_code_cells(code_cells):
    data_exploration_pattern = re.compile(r"(pd\.read|sns\.|plt\.)")
    data_preprocessing_pattern = re.compile(r"(train_test_split|StandardScaler|LabelEncoder)")
    model_initialization_pattern = re.compile(r"(tf\.keras\.models|torch\.nn\.|model = )")
    model_training_pattern = re.compile(r"(model\.fit|model\.train|epochs)")

    data_exploration_steps = []
    data_preprocessing_steps = []
    model_initialization_steps = []
    model_training_steps = []

    for cell in code_cells:
        if data_exploration_pattern.search(cell):
            data_exploration_steps.append(cell)
        elif data_preprocessing_pattern.search(cell):
            data_preprocessing_steps.append(cell)
        elif model_initialization_pattern.search(cell):
            model_initialization_steps.append(cell)
        elif model_training_pattern.search(cell):
            model_training_steps.append(cell)

    return {
        "data_exploration": data_exploration_steps,
        "data_preprocessing": data_preprocessing_steps,
        "model_initialization": model_initialization_steps,
        "model_training": model_training_steps
    }

if __name__ == "__main__":
    from extracted_cells import extract_cells
    notebook_path = '/Users/mm/Desktop/AI_code_generator/ALL_Classification_(Final_V8).ipynb'
    code_cells, _ = extract_cells(notebook_path)
    steps = preprocess_code_cells(code_cells)
    print("Data Exploration Steps:", steps["data_exploration"])
    print("Data Preprocessing Steps:", steps["data_preprocessing"])
    print("Model Initialization Steps:", steps["model_initialization"])
    print("Model Training Steps:", steps["model_training"])