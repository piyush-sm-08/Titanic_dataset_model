from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

def create_base_models():
    """
    Creates and returns the base models (Gradient Boosting and Random Forest).
    """

    gb = GradientBoostingClassifier(
        n_estimators=600,
        learning_rate=0.015,
        max_depth=3,
        subsample=0.9,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=4,
        random_state=42
    )
    
    return gb, rf

def create_stacking_model(gb, rf):
    """
    Creates and returns the StackingClassifier ensemble model.
    """

    stack = StackingClassifier(
        estimators=[
            ('gb', gb),
            ('rf', rf)
        ],
        final_estimator=LogisticRegression(max_iter=300),
        cv=7,           # cross-validation for better stacking performance  ..
        passthrough=True
    )
    return stack


if __name__ == '__main__':

    print("Testing model creation...")

    gb, rf = create_base_models()
    stack = create_stacking_model(gb, rf)

    print(stack)
