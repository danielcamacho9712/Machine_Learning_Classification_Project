import numpy as np
import pandas as pd

def feature_engineering(data):
    """ Perform feature engineering on the numerical input data. """

    df = data.copy()

    ### Create interaction features

    # May indicate if will rain or not (similar to precipitation)
    df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']

    # May indicate visibility conditions
    df['Wind_Speed_Precipitation_Interaction'] = df['Wind Speed'] * df['Precipitation (%)']

    # May indicate foggy conditions
    df['Humidity_Visibility_Interaction'] = df['Humidity'] / (df['Visibility (km)'] + 1)

    ### Rescale existing features

    # Some zeros in the feature, so it will be handled by log1p (Can help logistic regression model)
    df['UV_Index_log'] = np.log1p(df['UV Index'])
    df['Wind_Speed_log'] = np.log1p(df['Wind Speed'])
    df['Visibility_log'] = np.log1p(df['Visibility (km)'])

    # Some new categorical features based on existing numerical features (Can help tree-based models)
    df['Temperature_cat'] = pd.cut(
        df['Temperature'],
        bins=[-np.inf, 0, 15, 25, np.inf],
        labels=['Freezing', 'Cold', 'Moderate', 'Hot']
    )

    df['Humidity_cat'] = pd.cut(
        df['Humidity'],
        bins=[-np.inf, 30, 60, 100],
        labels=['Low', 'Medium', 'High']
    )

    df['Wind_Speed_cat'] = pd.cut(
        df['Wind Speed'],
        bins=[-np.inf, 10, 30, np.inf],
        labels=['Calm', 'Breezy', 'Windy']
    )

    return df


