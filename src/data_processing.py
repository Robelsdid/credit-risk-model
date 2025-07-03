# Placeholder for data processing and feature engineering functions 
 task-6

import pandas as pd

def process_data(data_path):
    df = pd.read_csv(data_path)
    return df, None 

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Create aggregate features per customer"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Group by CustomerId and calculate aggregates
        agg_features = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std'],
            'Value': ['sum', 'mean', 'count', 'std']
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = ['CustomerId'] + [
            f'Amount_{op}' for op in ['sum', 'mean', 'count', 'std']
        ] + [
            f'Value_{op}' for op in ['sum', 'mean', 'count', 'std']
        ]
        
        # Fill NaN values with 0 for count and appropriate values for others
        agg_features = agg_features.fillna({
            'Amount_count': 0,
            'Value_count': 0,
            'Amount_std': 0,
            'Value_std': 0
        })
        
        # Merge back to original dataframe
        X = X.merge(agg_features, on='CustomerId', how='left')
        
        return X

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extract datetime features from TransactionStartTime"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert TransactionStartTime to datetime
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Extract datetime features
        X['transaction_hour'] = X['TransactionStartTime'].dt.hour
        X['transaction_day'] = X['TransactionStartTime'].dt.day
        X['transaction_month'] = X['TransactionStartTime'].dt.month
        X['transaction_year'] = X['TransactionStartTime'].dt.year
        X['transaction_dayofweek'] = X['TransactionStartTime'].dt.dayofweek
        X['transaction_quarter'] = X['TransactionStartTime'].dt.quarter
        
        # Create cyclical features for hour and day of week
        X['hour_sin'] = np.sin(2 * np.pi * X['transaction_hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['transaction_hour'] / 24)
        X['dayofweek_sin'] = np.sin(2 * np.pi * X['transaction_dayofweek'] / 7)
        X['dayofweek_cos'] = np.cos(2 * np.pi * X['transaction_dayofweek'] / 7)
        
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables using Label Encoding"""
    
    def __init__(self, categorical_columns=None):
        self.categorical_columns = categorical_columns
        self.label_encoders = {}
        
    def fit(self, X, y=None):
        if self.categorical_columns is None:
            self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        self.categorical_columns = [col for col in self.categorical_columns if col != 'CustomerId']
        
        for col in self.categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.categorical_columns:
            if col in X.columns and col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        return X

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values using imputation"""
    
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = None
        
    def fit(self, X, y=None):
        # Select numerical columns for imputation
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X[numerical_cols])
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Handle numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.imputer is not None:
            X[numerical_cols] = self.imputer.transform(X[numerical_cols])
        
        # Handle categorical columns (fill with mode)
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if X[col].isnull().any():
                mode_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                X[col] = X[col].fillna(mode_value)
        
        return X

class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scale numerical features"""
    
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = None
        
    def fit(self, X, y=None):
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        if self.scaler is not None:
            self.scaler.fit(X[numerical_cols])
        
        return self
    
    def transform(self, X):
        X = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.scaler is not None:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X

def create_feature_engineering_pipeline():
    """Create the complete feature engineering pipeline"""
    
    pipeline = Pipeline([
        ('aggregate_features', AggregateFeatures()),
        ('datetime_features', DateTimeFeatures()),
        ('missing_values', MissingValueHandler(strategy='median')),
        ('categorical_encoder', CategoricalEncoder()),
        ('feature_scaler', FeatureScaler(method='standard'))
    ])
    
    return pipeline

def process_data(data_path, output_path=None):
    """Process the raw data through the feature engineering pipeline"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    
    # Create and fit pipeline
    print("Creating feature engineering pipeline...")
    pipeline = create_feature_engineering_pipeline()
    
    # Fit and transform
    print("Processing data...")
    df_processed = pipeline.fit_transform(df)
    print(f"Processed data shape: {df_processed.shape}")
    
    # Save processed data
    if output_path:
        df_processed.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
    
    return df_processed, pipeline

def calculate_rfm_metrics(df):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each CustomerId.
    Recency: Days since last transaction (relative to snapshot date)
    Frequency: Number of transactions
    Monetary: Total Amount spent
    """
    df = df.copy()
    # Ensure TransactionStartTime is datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    # Snapshot date is the day after the latest transaction
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    # Group by CustomerId
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'CustomerId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'CustomerId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()
    return rfm

def scale_rfm_features(rfm_df):
    """
    Scale the RFM features using StandardScaler.
    Returns a DataFrame with scaled features and the scaler object.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    rfm_scaled_df = rfm_df.copy()
    rfm_scaled_df[['Recency', 'Frequency', 'Monetary']] = rfm_scaled
    return rfm_scaled_df, scaler

def cluster_rfm_kmeans(rfm_scaled_df, n_clusters=3, random_state=42):
    """
    Cluster customers using K-Means on scaled RFM features.
    Returns the DataFrame with an added 'cluster' column and the fitted KMeans object.
    """
    features = rfm_scaled_df[['Recency', 'Frequency', 'Monetary']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_scaled_df = rfm_scaled_df.copy()
    rfm_scaled_df['cluster'] = kmeans.fit_predict(features)
    return rfm_scaled_df, kmeans

if __name__ == "__main__":
    
    input_path = "data/raw/data.csv"
    output_path = "data/processed/processed_data.csv"
    
    try:
        df_processed, pipeline = process_data(input_path, output_path)
        print("Feature engineering completed successfully!")
        
        # Show some statistics
        print("\nFeature engineering summary:")
        print(f"Original features: {len(pd.read_csv(input_path).columns)}")
        print(f"Processed features: {len(df_processed.columns)}")
        print(f"New features added: {len(df_processed.columns) - len(pd.read_csv(input_path).columns)}")
        
        # --- RFM TEST BLOCK ---
        print("\nTesting RFM calculation...")
        df_raw = pd.read_csv(input_path)
        rfm = calculate_rfm_metrics(df_raw)
        print(rfm.head())
        print(rfm.describe())
        
        # --- RFM SCALING TEST ---
        print("\nTesting RFM scaling...")
        rfm_scaled, _ = scale_rfm_features(rfm)
        print(rfm_scaled.head())
        print(rfm_scaled.describe())
        
        # --- RFM CLUSTERING TEST ---
        print("\nTesting RFM K-Means clustering...")
        rfm_clustered, kmeans = cluster_rfm_kmeans(rfm_scaled, n_clusters=3, random_state=42)
        print(rfm_clustered['cluster'].value_counts())
        print(rfm_clustered.head())
        
        # --- CLUSTER CENTROIDS (RFM MEANS) ---
        print("\nCluster RFM means (for interpretation):")
        print(rfm_clustered.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean())
        
        # --- ASSIGN HIGH-RISK LABEL ---
        print("\nAssigning high-risk label (is_high_risk=1 for cluster 0)...")
        rfm_clustered['is_high_risk'] = (rfm_clustered['cluster'] == 0).astype(int)
        print(rfm_clustered[['CustomerId', 'cluster', 'is_high_risk']].head(10))
        print(rfm_clustered['is_high_risk'].value_counts())
        
        # --- MERGE is_high_risk INTO PROCESSED DATASET ---
        print("\nMerging is_high_risk label into processed dataset and saving as processed_data_with_risk.csv ...")
        processed_path = "data/processed/processed_data.csv"
        processed_with_risk_path = "data/processed/processed_data_with_risk.csv"
        df_processed = pd.read_csv(processed_path)
        # Ensure CustomerId is string in both DataFrames
        df_processed['CustomerId'] = df_processed['CustomerId'].astype(str)
        rfm_clustered['CustomerId'] = rfm_clustered['CustomerId'].astype(str)
        df_merged = df_processed.merge(rfm_clustered[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        df_merged['is_high_risk'] = df_merged['is_high_risk'].fillna(0).astype(int)
        df_merged.to_csv(processed_with_risk_path, index=False)
        print(f"Merged data saved to: {processed_with_risk_path}")
        print(df_merged[['CustomerId', 'is_high_risk']].head(10))
        
    except Exception as e:
        print(f"Error during feature engineering: {e}")
 main
