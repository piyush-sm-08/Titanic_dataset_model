import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import create_preprocessor
from model import create_base_models, create_stacking_model
from utils import save_model 

def load_data(file_path):

    """Loads and preprocesses the Titanic dataset."""

    try:
        titanic = pd.read_csv(file_path)

    except FileNotFoundError:
        try:
             titanic = pd.read_csv('data/Titanic-Dataset.csv')

        except FileNotFoundError:
             try:
                 titanic = pd.read_csv('../data/Titanic-Dataset.csv')

             except FileNotFoundError:
                 raise FileNotFoundError("Could not find Titanic-Dataset.csv. Check the data/ folder location.")
    

    titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    

    titanic['Pclass'] = titanic['Pclass'].astype(int)
        
    # Split features and target
    x = titanic.drop(columns=['Survived'])
    y = titanic['Survived']
    
    return x, y

def main():

    data_file_path = 'data/Titanic-Dataset.csv'
    
    x, y = load_data(data_file_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x , y, test_size=0.2, random_state=42
    )

    preprocessor = create_preprocessor()
    gb, rf = create_base_models()
    stack = create_stacking_model(gb, rf)


    # Building Full Pipeline :
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', stack)
    ])

    # Train Model :
    print("Starting model training...")
    
    pipe.fit(x_train, y_train)

    print("Training complete.")

    # SAVE THE TRAINED MODEL ARTIFACT to models folder :
    model_filename = 'stacked_titanic_model.joblib'
    save_model(pipe, model_filename) 

    # 6. Predict and Evaluate
    y_pred = pipe.predict(x_test)

    print("\n            <--- Model Evaluation (StackingClassifier) --->")

    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}")

    print("\n Classification Report: ->>")

    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
    