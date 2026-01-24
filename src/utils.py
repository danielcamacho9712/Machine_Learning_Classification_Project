import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

