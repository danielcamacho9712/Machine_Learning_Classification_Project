import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    
    """Load data from a CSV file into a pandas DataFrame."""

    df = pd.read_csv(file_path)
    
    return df

def histogram_feature(data, bins = 30):
    """Plot histograms for all numerical features in the DataFrame."""

    numerical_cols = data.drop(columns=['Student_ID']).select_dtypes(include=['number']).columns

    for col in numerical_cols:

        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], 
                     bins=bins, 
                     kde=True)

        mean_val = data[col].mean()

        plt.axvline(mean_val, 
                    color='red', 
                    linestyle='dashed', 
                    linewidth=2)

        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

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
