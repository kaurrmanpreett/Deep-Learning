import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_data(filepath):
    """Load dataset and prepare for modeling."""
    # Load data
    df = pd.read_csv(filepath)
    
    # Separate target and features
    Y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    
    return x_train, x_test, y_train, y_test
