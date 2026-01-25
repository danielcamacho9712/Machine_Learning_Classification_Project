import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from src.preprocessing import build_preprocessor
from src.encoding import fit_and_save_label_encoder

def train_model(X_train, y_train):
    """ Train multiple models with appropriate preprocessing pipelines. """

    preprocessor_scalers, preprocessor_trees = build_preprocessor(X_train)

    models = {

        'log_reg': Pipeline(steps=[
            ('preprocessor', preprocessor_scalers),
            ('classifier', LogisticRegression(max_iter=500))
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
        model_pipeline.fit(X_train, y_train)
        trained_models[model_name] = model_pipeline

    return trained_models