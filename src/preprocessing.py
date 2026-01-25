from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(X):

    """Build a preprocessing pipeline for numerical and categorical features."""

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor_scalers = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    preprocessor_trees = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor_scalers, preprocessor_trees
