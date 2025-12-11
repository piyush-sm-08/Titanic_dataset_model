from pathlib import Path
import joblib 

def get_data_path(filename='Titanic-Dataset.csv'):
    """
    Calculates the absolute path to the data file in the 'data' directory
    relative to the 'src' directory.
    """

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'
    file_path = data_dir / filename

    return file_path

def get_model_path(filename='stacked_titanic_model.joblib'):
    """
    Calculates the absolute path to save the model in the 'models' directory.
    """

    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent / 'models'

    # Ensure the models directory exists
    
    model_dir.mkdir(exist_ok=True) 
    file_path = model_dir / filename

    return file_path

def save_model(model, filename):
    """Saves the trained model object using joblib."""

    file_path = get_model_path(filename)
    joblib.dump(model, file_path)

    print(f"\nModel successfully saved to: {file_path}")

def load_model(filename):
    """Loads a trained model object using joblib."""

    file_path = get_model_path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found at {file_path}. Please train the model first.")
    
    model = joblib.load(file_path)

    print(f"\nModel successfully loaded from: {file_path}")

    return model