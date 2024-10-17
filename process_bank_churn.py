import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Dict, Optional, Any

def split_data(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Split the DataFrame into training and validation sets.
    
    Args:
        df (pd.DataFrame): The raw DataFrame.
        target_col (str): The name of the target column.
        
    Returns:
        Dict[str, pd.DataFrame]: A dictionary with training and validation DataFrames.
    """
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    return {'train_df': train_df, 'val_df': val_df}

def get_input_output(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Separate input features and target column.
    
    Args:
        df (pd.DataFrame): The DataFrame.
        target_col (str): The name of the target column.
        
    Returns:
        Dict[str, pd.DataFrame]: A dictionary with inputs and targets DataFrames.
    """
    input_cols = [col for col in df.columns if col != target_col]
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return {'inputs': inputs, 'targets': targets, 'input_cols': input_cols}

def create_transformers(
    numeric_cols: List[str], 
    categorical_cols: List[str], 
    scaler_numeric: bool = True
) -> ColumnTransformer:
    """
    Create ColumnTransformer for preprocessing the data.
    
    Args:
        numeric_cols (List[str]): List of numeric columns.
        categorical_cols (List[str]): List of categorical columns.
        scaler_numeric (bool): Whether to scale numeric columns.
        
    Returns:
        ColumnTransformer: The preprocessor with the specified transformers.
    """
    transformers = []
    if scaler_numeric:
        numeric_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
    else:
        numeric_transformer = 'passthrough'
    
    transformers.append(('num', numeric_transformer, numeric_cols))
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    transformers.append(('cat', categorical_transformer, categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get the feature names after transformation.
    
    Args:
        preprocessor (ColumnTransformer): The preprocessor.
        
    Returns:
        List[str]: The feature names.
    """
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num' and transformer != 'passthrough':
            feature_names += columns
        elif name == 'cat':
            encoder = transformer.named_steps['onehot']
            feature_names += encoder.get_feature_names_out(columns).tolist()
        elif name == 'num' and transformer == 'passthrough':
            feature_names += columns  # add original column names for passthrough
    return feature_names

def preprocess_data(
    raw_df: pd.DataFrame, 
    target_col: str, 
    scaler_numeric: bool = True, 
    numeric_cols: Optional[List[str]] = None, 
    categorical_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Preprocess the raw DataFrame.
    
    Args:
        raw_df (pd.DataFrame): The raw DataFrame.
        target_col (str): The name of the target column.
        scaler_numeric (bool): Whether to scale numeric columns.
        numeric_cols (Optional[List[str]]): List of numeric columns.
        categorical_cols (Optional[List[str]]): List of categorical columns.
        
    Returns:
        Dict[str, Any]: A dictionary with preprocessed data and transformers.
    """
    data_splits = split_data(raw_df, target_col)
    train_data = get_input_output(data_splits['train_df'], target_col)
    val_data = get_input_output(data_splits['val_df'], target_col)
    
    if numeric_cols is None:
        numeric_cols = train_data['inputs'].select_dtypes(include=np.number).columns.tolist()
    if categorical_cols is None:
        categorical_cols = train_data['inputs'].select_dtypes(include='object').columns.tolist()
    
    preprocessor = create_transformers(numeric_cols, categorical_cols, scaler_numeric)
    
    X_train = preprocessor.fit_transform(train_data['inputs'])
    X_val = preprocessor.transform(val_data['inputs'])
    
    # Get feature names for the resulting dataframe
    feature_names = get_feature_names(preprocessor)

    X_train_df = pd.DataFrame(X_train, columns=feature_names, index=train_data['inputs'].index)
    X_val_df = pd.DataFrame(X_val, columns=feature_names, index=val_data['inputs'].index)
    
    return {
        'X_train': X_train_df,
        'train_targets': train_data['targets'],
        'X_val': X_val_df,
        'val_targets': val_data['targets'],
        'preprocessor': preprocessor,
        'input_cols': feature_names
    }

def preprocess_new_data(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Preprocess new data using the fitted preprocessor.
    
    Args:
        df (pd.DataFrame): The new data.
        preprocessor (ColumnTransformer): The fitted preprocessor.
        
    Returns:
        pd.DataFrame: The transformed data as a DataFrame.
    """
    transformed_data = preprocessor.transform(df)
    
    # Get feature names for the resulting dataframe
    feature_names = get_feature_names(preprocessor)
    
    return pd.DataFrame(transformed_data, columns=feature_names, index=df.index)