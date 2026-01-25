from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def cross_validate_final_model(X, y, preprocessor, n_splits=5):
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
        'roc_auc': 'roc_auc_ovr_weighted'
    }

    scores = cross_validate(
        pipeline,
        X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    results = pd.DataFrame({
        metric: scores[f'test_{metric}']
        for metric in scoring
    })

    results['fold'] = np.arange(1, n_splits + 1)

    return results
