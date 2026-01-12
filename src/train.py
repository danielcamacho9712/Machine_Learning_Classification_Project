import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor

def train_model(data):

    X = data.drop(columns=['Placement_Status', 'Student_ID'])
    y = data['Placement_Status']

    preprocessor_scalers, preprocessor_trees = build_preprocessor(X)

    models = {

        'log_reg': Pipeline(steps=[
            ('preprocessor', preprocessor_scalers),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),

        'rand_forest': Pipeline(steps=[
            ('preprocessor', preprocessor_trees),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),

        'grad_boost': Pipeline(steps=[
            ('preprocessor', preprocessor_trees),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
    }

    trained_models = {}

    for model_name, model_pipeline in models.items():
        model_pipeline.fit(X, y)
        trained_models[model_name] = model_pipeline

    return trained_models