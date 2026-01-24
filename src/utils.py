import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    
    """Load data from a CSV file into a pandas DataFrame."""

    df = pd.read_csv(file_path)
    
    return df

def histogram_feature(data, bins = 30):
    """Plot histograms for all numerical features in the DataFrame."""

    numerical_cols = data.select_dtypes(include=['number']).columns

    data[numerical_cols].hist(bins=bins, figsize=(15, 8))

    return

def correlation_matrix(data):
    """Plot the correlation matrix for numerical features in the DataFrame."""

    numerical_cols = data.select_dtypes(include=['number']).columns
    corr = data[numerical_cols].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(corr,
                annot=True, 
                fmt=".2f", 
                square=True)
    
    plt.title('Correlation Matrix')
    plt.show()

    return

def categorical_cardinality(data):
    """Cardinality of categorical features in the DataFrame."""

    cardinalities = {}

    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        cardinalities[col] = data[col].nunique()

    return cardinalities

def compute_mutual_info(data):
    """Compute mutual information between features and the target variable."""

    X = data.drop(columns=['Weather Type'])
    y = data['Weather Type']

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    discrete_features = [True if col in categorical_cols else False for col in X.columns]

    mi = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi_series