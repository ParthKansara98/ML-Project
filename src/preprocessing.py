import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(path):
    df = pd.read_csv(path)
    # Replace 0 with NaN for specific columns only (excluding 'Outcome')
    cols_to_fix = [col for col in df.columns if col != 'Outcome']
    df[cols_to_fix] = df[cols_to_fix].replace(0, pd.NA)
    df.fillna(df.mean(), inplace=True)
    return df

def split_and_scale_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
