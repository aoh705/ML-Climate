import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import time
import json
import os
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import warnings

# Filter out specific warnings
warnings.filterwarnings("ignore", message="Input array is not float32; it has been recast to float32")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Import TreeFFuser if available
try:
    from treeffuser import Treeffuser
    TREEFFUSER_AVAILABLE = True
except ImportError:
    TREEFFUSER_AVAILABLE = False
    print("TreeFFuser not available. Will use RandomForest.")

class DroughtPredictor:
    """
    A class to predict drought indices using TreeFFuser (or RandomForest).
    
    This class handles data preprocessing, feature engineering, model training,
    and prediction for drought indices D0-D4 using historical climate data.
    """
    
    def __init__(self, min_year=1950, max_year=2013, result_dir='results', data_simul=None):
        """
        Initialize the DroughtPredictor.
        
        Parameters:
        min_year : int
            Minimum year to include in historical data
        max_year : int
            Maximum year to include in historical data
        result_dir : str
            Directory to save results and visualizations
        data_simul : str, optional
            Simulation scenario (e.g., 'wd85')
        """
        # Configuration parameters
        self.min_year = min_year
        self.max_year = max_year
        self.result_dir = result_dir
        self.target_cols = ['D0', 'D1', 'D2', 'D3', 'D4']
        self.lag_periods = [1, 2, 3, 6, 12, 24, 36]
        # , 6, 12, 24, 36
        
        # Internal storage
        self.historical_df = None
        self.future_df = None
        self.historical_df_features = None
        self.abrv = data_simul
        self.future_df_features = None
        self.monthly_means = None
        self.monthly_stds = None
        self.all_feature_cols = None
        self.selected_features = {}
        self.cv_results = None
        self.d4_binary_results = None
        self.avg_results = None
        self.avg_d4_binary = None
        self.final_models = {}
        self.final_scalers = {}
        self.classifier = None
        self.binary_scaler = None
        self.future_predictions = None
        
        # Create result directory
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Performance tracking
        self.start_time = time.time()
    
    def print_section(self, title):
        """Print a section header to make output more readable"""
        print("\n" + "="*80)
        print(f" {title} ".center(80, "="))
        print("="*80)
    
    def format_date(self, date_str):
        """Standardize date formats from both datasets"""
        if len(date_str) <= 7:  # Format like '2025-01'
            return f"{date_str}-01"  # Add day to match historical format
        return date_str  # Already in full format
    
    def load_data(self, historical_file, future_file):
        """
        Load historical and future data from CSV files.
        
        Parameters:
        historical_file : str
            Path to historical data CSV
        future_file : str
            Path to future data CSV
            
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        self.print_section("LOADING AND PREPARING DATA")
        
        # Load data files
        self.historical_df = pd.read_csv(historical_file)
        self.future_df_csv = future_file
        self.future_df = pd.read_csv(future_file)
        
        # Convert and standardize dates
        self.historical_df['date'] = pd.to_datetime(self.historical_df['date'])
        self.future_df['date'] = pd.to_datetime(self.future_df['date'].apply(self.format_date))

        # # FOR TESTING PURPOSES
        # print("\n FOR TESTING PURPOSES: DOING SUBSET OF DATA")
        # self.future_df = self.future_df[self.future_df['date'] < '2026-12-01']
        
        # Ensure historical data is sorted by date
        self.historical_df = self.historical_df.sort_values('date')
        
        # Filter historical data
        self.historical_df = self.historical_df[
            (self.historical_df['date'].dt.year >= self.min_year) & 
            (self.historical_df['date'].dt.year <= self.max_year)
        ]
        
        print(f"Historical data: {self.historical_df.shape[0]} rows from {self.historical_df['date'].min()} to {self.historical_df['date'].max()}")
        print(f"Future data: {self.future_df.shape[0]} rows from {self.future_df['date'].min()} to {self.future_df['date'].max()}")
        
        # Extract month and year as separate features
        for df in [self.historical_df, self.future_df]:
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            # Convert month to a cyclical feature
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            
            # Add quarterly seasonal indicators
            df['quarter'] = ((df['month'] - 1) // 3) + 1
            for q in range(1, 5):
                df[f'quarter_{q}'] = (df['quarter'] == q).astype(int)
        
        # Pre-calculate monthly patterns
        self.monthly_means = {col: self.historical_df.groupby('month')[col].mean() 
                             for col in self.target_cols}
        self.monthly_stds = {col: self.historical_df.groupby('month')[col].std() 
                            for col in self.target_cols}
        
        return self
    
    def create_lag_features(self, df, target_cols, lag_periods=None, fill_method='monthly_mean'):
        """
        Create enhanced lag features for the target columns.
        
        Parameters:
        df : DataFrame
            DataFrame with time-indexed data
        target_cols : list
            List of columns to create features for (D0-D4)
        lag_periods : list, optional
            List of lag periods in months
        fill_method : str
            Method to fill missing values ('monthly_mean' or 'forward_fill')
            
        Returns:
        DataFrame
            DataFrame with added features
        """
        if lag_periods is None:
            lag_periods = self.lag_periods
            
        df = df.copy()
        
        # Ensure the dataframe is sorted by date
        df = df.sort_values('date')
        
        # Pre-calculate monthly means for filling
        if fill_method == 'monthly_mean':
            monthly_means = {col: df.groupby('month')[col].mean() 
                            for col in target_cols if col in df.columns}
        
        # Create lag features for each target column
        for col in target_cols:
            if col in df.columns:
                for lag in lag_periods:
                    lag_col = f'{col}_lag_{lag}'
                    df[lag_col] = df[col].shift(lag)
                    
                    # Fill missing values
                    if fill_method == 'monthly_mean' and lag_col in df.columns:
                        # Use vectorized operations for faster filling
                        mask = df[lag_col].isna()
                        if mask.any():
                            df.loc[mask, lag_col] = df.loc[mask, 'month'].map(monthly_means[col])
                    elif fill_method == 'forward_fill':
                        df[lag_col] = df[lag_col].ffill()
        
        # Create interaction features between meteorological variables
        for i, var1 in enumerate(['airtemp', 'rainfall', 'sm']):
            for j, var2 in enumerate(['baseflow', 'ev', 'runoff']):
                if i != j and var1 in df.columns and var2 in df.columns:
                    df[f'{var1}_{var2}_interaction'] = df[var1] * df[var2]
        
        # Add rolling mean features for climate variables
        for col in ['airtemp', 'rainfall', 'sm']:
            if col in df.columns:
                for window in [3, 6, 12]:
                    df[f'{col}_roll_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Add seasonal lag features (same month in previous years)
        for col in target_cols:
            if col in df.columns:
                for years_back in [1, 2, 3]:
                    months_back = years_back * 12
                    seasonal_lag_col = f'{col}_seasonal_lag_{years_back}y'
                    df[seasonal_lag_col] = df[col].shift(months_back)
                    
                    # Fill missing values for seasonal lags
                    if fill_method == 'monthly_mean':
                        mask = df[seasonal_lag_col].isna()
                        if mask.any():
                            df.loc[mask, seasonal_lag_col] = df.loc[mask, 'month'].map(monthly_means[col])
        
        return df
    
    def prepare_features(self):
        """
        Apply feature engineering to historical data.
        
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        self.print_section("ENHANCED FEATURE ENGINEERING")
        
        # Apply enhanced feature engineering to historical data
        self.historical_df_features = self.create_lag_features(
            self.historical_df, self.target_cols, self.lag_periods
        )
        
        # Print descriptive statistics of target columns
        print("\nTarget variable statistics:")
        for col in self.target_cols:
            stats = self.historical_df[col].describe()
            print(f"{col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, median={stats['50%']:.2f}")
        
        # Create D4 binary classification target
        self.historical_df_features['D4_binary'] = (self.historical_df_features['D4'] > 0).astype(int)
        print(f"\nD4 binary class distribution: {self.historical_df_features['D4_binary'].value_counts()}")
        
        # Check if any NaN values remain
        nan_count = self.historical_df_features.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values remain in the feature-engineered historical data")
            print(self.historical_df_features.isna().sum())
            # Drop rows with remaining NaN values as a last resort
            self.historical_df_features = self.historical_df_features.dropna()
            print(f"Dropped rows with NaN values. Remaining rows: {len(self.historical_df_features)}")
        else:
            print("No NaN values in the feature-engineered historical data")
        
        # Define feature columns
        base_feature_cols = [col for col in ['airtemp', 'baseflow', 'ev', 'rainfall', 'runoff', 'sm', 'snowfall', 'snowwater'] 
                            if col in self.historical_df_features.columns]
        
        seasonal_features = ['month_sin', 'month_cos', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
        rolling_features = [col for col in self.historical_df_features.columns if '_roll_' in col]
        interaction_features = [col for col in self.historical_df_features.columns if '_interaction' in col]
        derived_feature_cols = [col for col in self.historical_df_features.columns 
                               if (('_lag_' in col or '_seasonal_lag_' in col) and 
                                  col.split('_')[0] in self.target_cols)]
        
        # Combine all feature columns
        self.all_feature_cols = base_feature_cols + seasonal_features + rolling_features + interaction_features + derived_feature_cols
        
        print(f"Total number of features: {len(self.all_feature_cols)}")
        print(f"Base features: {len(base_feature_cols)}")
        print(f"Seasonal features: {len(seasonal_features)}")
        print(f"Rolling features: {len(rolling_features)}")
        print(f"Interaction features: {len(interaction_features)}")
        print(f"Derived features: {len(derived_feature_cols)}")
        
        return self
    
    def analyze_feature_importance(self, X, y, feature_names, n_top=20):
        """
        Analyze feature importance using Random Forest.
        
        Parameters:
        X : DataFrame or array
            Feature matrix
        y : Series or array
            Target vector
        feature_names : list
            List of feature names
        n_top : int
            Number of top features to print
            
        Returns:
        tuple
            Sorted feature indices and their importance values
        """
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print top features
        print("\nTop important features:")
        for i in range(min(n_top, len(feature_names))):
            print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return indices, importances
    
    def select_features(self):
        """
        Run feature importance analysis and select features for each target.
        
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        self.print_section("FEATURE IMPORTANCE ANALYSIS")
        
        # Prepare data for feature importance analysis
        X_importance = self.historical_df_features[self.all_feature_cols]
        feature_importance_results = {}
        self.selected_features = {}
        
        # Run feature importance for each target and select top features
        for col in self.target_cols:
            print(f"\nFeature importance for {col}:")
            y_target = self.historical_df_features[col]
            indices, importances = self.analyze_feature_importance(X_importance, y_target, self.all_feature_cols)
            feature_importance_results[col] = (indices, importances)
            
            # Select top features for each target (adaptive feature selection)
            n_features = 30  # Maximum number of features to use
            # For D3 which had negative R², we'll use more selective feature selection
            if col == 'D3':
                n_features = 20
            
            # Select features with importance above a threshold
            threshold = 0.005  # Minimum importance to include feature
            important_indices = [i for i, imp in enumerate(importances) if imp > threshold][:n_features]
            self.selected_features[col] = [self.all_feature_cols[i] for i in important_indices]
            
            print(f"Selected {len(self.selected_features[col])} features for {col}")
        
        return self
    
    def train_treeffuser_model(self, X, y, col_name):
        """
        Train a TreeFFuser model with proper handling of data types.
        
        Parameters:
        X : DataFrame or array
            Feature matrix
        y : Series or array
            Target vector
        col_name : str
            Name of the target column (for logging)
            
        Returns:
        model : Treeffuser or RandomForestRegressor
            Trained model
        """
        print(f"Training model for {col_name}...")
        
        # Store feature names if available
        feature_names = None
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        
        # Convert to float32 to avoid warnings
        X = np.asarray(X, dtype=np.float32)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Convert y to float32 as well
        y = np.asarray(y, dtype=np.float32)
        
        # Make sure y is 2D for TreeFFuser (reshape if needed)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Determine optimal n_estimators based on target importance
        if col_name in ['D0', 'D1', 'D4']:
            n_estimators = 150  # Important targets get more estimators
        else:
            n_estimators = 100  # Less critical targets get fewer estimators
        
        # Create and train TreeFFuser model
        if TREEFFUSER_AVAILABLE:
            try:
                model = Treeffuser(n_estimators=n_estimators, seed=42)
                
                # Store feature names with the model
                if hasattr(model, 'feature_names_in_') and feature_names is not None:
                    model.feature_names_in_ = feature_names
                    
                model.fit(X, y)
                print(f"  Successfully trained TreeFFuser with {n_estimators} estimators")
                return model
            except Exception as e:
                print(f"  Error training TreeFFuser model: {str(e)}")
                print("  Falling back to RandomForest model")
        else:
            print("  Using RandomForest (TreeFFuser not available)")
            
        # Fall back to RandomForest
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        
        # Convert y back to 1D for RandomForest if needed
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.ravel()
            
        rf_model.fit(X, y)
        return rf_model
    
    def cross_validate(self, n_splits=5):
        """
        Perform time series cross-validation.
        
        Parameters:
        n_splits : int
            Number of CV splits
            
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        self.print_section("TIME SERIES CROSS-VALIDATION")
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Get unique years for splitting
        years = self.historical_df_features['year'].unique()
        years.sort()
        
        # Calculate approximate split years
        split_years = []
        year_interval = (years[-1] - years[0]) // n_splits
        for i in range(n_splits - 1):
            split_year = years[0] + (i + 1) * year_interval
            split_years.append(split_year)
        
        print(f"Time series cross-validation with {n_splits} splits")
        print(f"Data from {years[0]} to {years[-1]}")
        print(f"Approximate split years: {split_years}")
        
        # Initialize dictionaries to store CV results
        self.cv_results = {
            'r2': {col: [] for col in self.target_cols},
            'mae': {col: [] for col in self.target_cols},
            'mse': {col: [] for col in self.target_cols},
            'rmse': {col: [] for col in self.target_cols}
        }
        
        self.d4_binary_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Execute time series cross-validation
        for i, (train_idx, test_idx) in enumerate(tscv.split(self.historical_df_features)):
            print(f"\nFold {i+1}/{n_splits}")
            
            # Extract train and test sets
            train_df = self.historical_df_features.iloc[train_idx]
            test_df = self.historical_df_features.iloc[test_idx]
            
            train_years = (train_df['year'].min(), train_df['year'].max())
            test_years = (test_df['year'].min(), test_df['year'].max())
            
            print(f"Train period: {train_years[0]}-{train_years[1]} ({len(train_df)} samples)")
            print(f"Test period: {test_years[0]}-{test_years[1]} ({len(test_df)} samples)")
            
            # Train and evaluate models for each target
            for col in self.target_cols:
                # Use selected features for each target
                X_train = train_df[self.selected_features[col]]
                y_train = train_df[col]
                X_test = test_df[self.selected_features[col]]
                y_test = test_df[col]
                
                # Scale the features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = self.train_treeffuser_model(X_train_scaled, y_train, col)
                
                # Make predictions - handle different model types
                if TREEFFUSER_AVAILABLE and isinstance(model, Treeffuser):
                    # For TreeFFuser: use sample mean
                    samples = model.sample(X_test_scaled, n_samples=150)
                    y_pred = samples.mean(axis=0)
                    # If y_pred is 2D with one column, convert to 1D
                    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                        y_pred = y_pred.ravel()
                else:
                    # For RandomForest fallback
                    y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store results
                self.cv_results['r2'][col].append(r2)
                self.cv_results['mae'][col].append(mae)
                self.cv_results['mse'][col].append(mse)
                self.cv_results['rmse'][col].append(rmse)
                
                print(f"{col} results: R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
            
            # Train and evaluate D4 binary classification model
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            X_train_bin = train_df[self.selected_features['D4']]
            y_train_bin = train_df['D4_binary']
            X_test_bin = test_df[self.selected_features['D4']]
            y_test_bin = test_df['D4_binary']
            
            # Scale features
            scaler_bin = StandardScaler()
            X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
            X_test_bin_scaled = scaler_bin.transform(X_test_bin)
            
            # Train binary classifier for D4
            classifier = RandomForestClassifier(
                n_estimators=150, 
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',  # Handle class imbalance
                random_state=42,
                n_jobs=-1
            )
            classifier.fit(X_train_bin_scaled, y_train_bin)
            
            # Make predictions
            y_pred_bin = classifier.predict(X_test_bin_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_bin, y_pred_bin)
            precision = precision_score(y_test_bin, y_pred_bin, zero_division=0)
            recall = recall_score(y_test_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
            
            # Store results
            self.d4_binary_results['accuracy'].append(accuracy)
            self.d4_binary_results['precision'].append(precision)
            self.d4_binary_results['recall'].append(recall)
            self.d4_binary_results['f1'].append(f1)
            
            print(f"D4 Binary Classification: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
            print(f"Precision = {precision:.4f}, Recall = {recall:.4f}")
        
        # Calculate average CV metrics
        self.avg_results = {
            'r2': {col: np.mean(self.cv_results['r2'][col]) for col in self.target_cols},
            'mae': {col: np.mean(self.cv_results['mae'][col]) for col in self.target_cols},
            'mse': {col: np.mean(self.cv_results['mse'][col]) for col in self.target_cols},
            'rmse': {col: np.mean(self.cv_results['rmse'][col]) for col in self.target_cols}
        }
        
        self.avg_d4_binary = {
            'accuracy': np.mean(self.d4_binary_results['accuracy']),
            'precision': np.mean(self.d4_binary_results['precision']),
            'recall': np.mean(self.d4_binary_results['recall']),
            'f1': np.mean(self.d4_binary_results['f1'])
        }
        
        # Print average results
        print("\nAverage Cross-Validation Results:")
        for col in self.target_cols:
            print(f"{col}: R² = {self.avg_results['r2'][col]:.4f}, MAE = {self.avg_results['mae'][col]:.4f}, RMSE = {self.avg_results['rmse'][col]:.4f}")
        
        print(f"\nD4 Binary Classification: Accuracy = {self.avg_d4_binary['accuracy']:.4f}, F1 = {self.avg_d4_binary['f1']:.4f}")
        print(f"Precision = {self.avg_d4_binary['precision']:.4f}, Recall = {self.avg_d4_binary['recall']:.4f}")
        
        # Save metrics to JSON
        metrics = {
            'regression': {
                'drought_indices': {col: {
                    'r2': float(self.avg_results['r2'][col]),
                    'mae': float(self.avg_results['mae'][col]),
                    'mse': float(self.avg_results['mse'][col]),
                    'rmse': float(self.avg_results['rmse'][col])
                } for col in self.target_cols}
            },
            'classification': {
                'D4_binary': {
                    'accuracy': float(self.avg_d4_binary['accuracy']),
                    'precision': float(self.avg_d4_binary['precision']),
                    'recall': float(self.avg_d4_binary['recall']),
                    'f1': float(self.avg_d4_binary['f1'])
                }
            },
            'model_details': {
                'time_period': f"{self.min_year}-{self.max_year}",
                'n_splits': n_splits,
                'features_count': {col: len(self.selected_features[col]) for col in self.target_cols}
            }
        }
        # Create directories before saving
        output_dir = f'{self.result_dir}/{self.abrv}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to file
        with open(f'{self.result_dir}/{self.abrv}/{self.abrv}_performance_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nPerformance metrics saved to '{self.result_dir}/performance_metrics.json'")
        
        return self
    
    def train_final_models(self):
        """
        Train final models on the entire dataset.
        
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        self.print_section("FINAL MODEL TRAINING")
        
        # Train regression models
        for col in self.target_cols:
            # Use selected features for each target
            X_all = self.historical_df_features[self.selected_features[col]]
            y_all = self.historical_df_features[col]
            
            # Scale features
            scaler = StandardScaler()
            X_all_scaled = scaler.fit_transform(X_all)
            
            # Train model
            print(f"Training final model for {col}...")
            model = self.train_treeffuser_model(X_all_scaled, y_all, col)
            
            # Store model and scaler
            self.final_models[col] = model
            self.final_scalers[col] = scaler
        
        # Train binary classification model for D4
        print("Training final D4 binary classification model...")
        X_all_bin = self.historical_df_features[self.selected_features['D4']]
        y_all_bin = self.historical_df_features['D4_binary']
        
        # Scale features
        self.binary_scaler = StandardScaler()
        X_all_bin_scaled = self.binary_scaler.fit_transform(X_all_bin)
        
        # Train model
        self.classifier = RandomForestClassifier(
            n_estimators=150, 
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_all_bin_scaled, y_all_bin)
        
        return self
    
    def make_predictions(self):
        """
        Make autoregressive predictions with domain knowledge constraints.
        
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        self.print_section("MAKING PREDICTIONS")

        # Add these lines at the beginning of make_predictions method:
        print("Type of self.future_df_csv:", type(self.future_df_csv))
        print("Is self.future_df a DataFrame?", isinstance(self.future_df, pd.DataFrame))
        
        print("\nStarting autoregressive prediction with domain knowledge...")
        
        # Create working copies
        historical_df = self.historical_df.copy()
        future_df = self.future_df.copy()
        
        # Add target columns to future_df
        for col in self.target_cols:
            if col not in future_df.columns:
                future_df[col] = np.nan
        
        # Combine historical and future data
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        print(f"Combined data shape: {combined_df.shape}")
        print(f"Prediction period: {future_df['date'].min()} to {future_df['date'].max()}")
        
        # Get index where future data starts
        future_start_idx = combined_df[combined_df['date'] >= future_df['date'].min()].index[0]
        print(f"Future data starts at index {future_start_idx}")
        
        # Pre-calculate monthly means and stds for constraints
        monthly_means = self.monthly_means
        monthly_stds = self.monthly_stds
        
        # Create temperature anomaly as a proxy for ENSO
        combined_df['temp_anomaly'] = combined_df.groupby('month')['airtemp'].transform(
            lambda x: x - x.mean()
        )
        combined_df['enso_effect'] = np.tanh(combined_df['temp_anomaly'])  # Bounded effect
        
        # Dictionary to store prediction statistics
        pred_stats = {col: [] for col in self.target_cols}
        
        # Process future data with autoregressive approach
        for i in range(future_start_idx, len(combined_df)):
            current_date = combined_df.loc[i, 'date']
            current_month = combined_df.loc[i, 'month']
            print(f"Predicting for {current_date.strftime('%Y-%m')}...", end=" ")
            
            # Update features based on previous predictions
            current_df = self.create_lag_features(
                combined_df.iloc[:i+1], 
                self.target_cols, 
                self.lag_periods,
                fill_method='monthly_mean'
            )
            
            # Get the current row for prediction
            current_row = current_df.iloc[-1:]
            
            # Make predictions for each target
            for col in self.target_cols:
                try:
                    # Extract features for this target
                    model_features = [f for f in self.selected_features[col] if f in current_row.columns]
                    
                    # Extract feature values
                    features = current_row[model_features].copy()
                    
                    # Scale the features
                    scaler = self.final_scalers[col]
                    features_scaled = scaler.transform(features)
                    
                    # Get the model
                    model = self.final_models[col]
                    
                    # Make prediction
                    if TREEFFUSER_AVAILABLE and isinstance(model, Treeffuser):
                        # For TreeFFuser: use sample mean
                        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                            # Create DataFrame with proper feature names
                            features_df = pd.DataFrame(features_scaled, columns=model.feature_names_in_)
                            samples = model.sample(features_df, n_samples=150)
                        else:
                            # Fall back to using the array directly
                            samples = model.sample(features_scaled.astype(np.float32), n_samples=150)
                            
                        prediction = samples.mean(axis=0)[0]
                    else:
                        # For RandomForest fallback
                        prediction = model.predict(features_scaled)[0]
                    
                    # Apply constraints to ensure realistic predictions
                    min_val = max(0, historical_df[col].min())
                    max_val = min(100, historical_df[col].max())
                    
                    # Domain knowledge: Adjust for climate pattern effects like ENSO
                    enso_effect = combined_df.loc[i, 'enso_effect']
                    enso_adjustment = enso_effect * monthly_stds[col][current_month] * 0.5
                    prediction += enso_adjustment
                    
                    # Domain knowledge: Ensure drought hierarchy D0 >= D1 >= D2 >= D3 >= D4
                    if col == 'D1' and 'D0' in combined_df.columns:
                        prediction = min(prediction, combined_df.loc[i, 'D0'])
                    elif col == 'D2' and 'D1' in combined_df.columns:
                        prediction = min(prediction, combined_df.loc[i, 'D1'])
                    elif col == 'D3' and 'D2' in combined_df.columns:
                        prediction = min(prediction, combined_df.loc[i, 'D2'])
                    elif col == 'D4' and 'D3' in combined_df.columns:
                        prediction = min(prediction, combined_df.loc[i, 'D3'])
                    
                    # Additional constraint: prediction should not change too dramatically
                    if i > 0 and not pd.isna(combined_df.loc[i-1, col]):
                        prev_value = combined_df.loc[i-1, col]
                        month_std = monthly_stds[col][current_month]
                        
                        # Limit change based on seasonal volatility
                        max_change = 2.5 * month_std
                        prediction = max(prev_value - max_change, min(prev_value + max_change, prediction))
                    
                    # Final bounds check
                    prediction = max(min_val, min(max_val, prediction))
                    
                    # Store the prediction
                    combined_df.loc[i, col] = prediction
                    
                    # Store prediction value for statistics (handle arrays)
                    if hasattr(prediction, '__len__') and not isinstance(prediction, (str, bytes)):
                        pred_stats[col].append(float(prediction[0]))
                    else:
                        pred_stats[col].append(float(prediction))
                    
                except Exception as e:
                    print(f"\nError predicting {col}: {str(e)}")
                    # Use fallback value
                    if i > 0 and not pd.isna(combined_df.loc[i-1, col]):
                        fallback_value = combined_df.loc[i-1, col]
                    else:
                        fallback_value = monthly_means[col][current_month]
                    
                    combined_df.loc[i, col] = fallback_value
                    pred_stats[col].append(float(fallback_value))
            
            # Make D4 binary prediction
            try:
                # Extract features for D4 binary classifier
                d4_features = current_row[self.selected_features['D4']].copy()
                
                # Scale features
                d4_features_scaled = self.binary_scaler.transform(d4_features)
                
                # Make binary prediction
                d4_binary_pred = self.classifier.predict(d4_features_scaled)[0]
                d4_binary_prob = self.classifier.predict_proba(d4_features_scaled)[0][1]
                
                # Adjust D4 regression based on binary prediction
                if d4_binary_pred == 1 and combined_df.loc[i, 'D4'] < 1.0:
                    combined_df.loc[i, 'D4'] = max(combined_df.loc[i, 'D4'], 1.0)
                
                # Store binary prediction
                combined_df.loc[i, 'D4_binary_pred'] = d4_binary_pred
                combined_df.loc[i, 'D4_binary_prob'] = d4_binary_prob
                
            except Exception as e:
                print(f"\nError in D4 binary classification: {str(e)}")
                # Use regression-based classification as fallback
                combined_df.loc[i, 'D4_binary_pred'] = 1 if combined_df.loc[i, 'D4'] > 0 else 0
                combined_df.loc[i, 'D4_binary_prob'] = min(1.0, combined_df.loc[i, 'D4'] / 5.0)
            
            print("Done")
        
        # Return only the future predictions
        self.future_predictions = combined_df[combined_df['date'] >= future_df['date'].min()].copy()
        
        # Print prediction statistics
        print("\nPrediction statistics by target:")
        for col in self.target_cols:
            try:
                # Calculate statistics on the scalar values
                stats = {
                    'mean': np.mean(pred_stats[col]),
                    'std': np.std(pred_stats[col]),
                    'min': np.min(pred_stats[col]),
                    'max': np.max(pred_stats[col])
                }
                print(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
            except Exception as e:
                # Fallback statistics based on the final predictions
                print(f"Error calculating statistics for {col}: {str(e)}")
                col_stats = self.future_predictions[col].describe()
                print(f"{col}: mean={col_stats['mean']:.2f}, std={col_stats['std']:.2f}, min={col_stats['min']:.2f}, max={col_stats['max']:.2f}")
        
        # Final check for any remaining NaN values
        for col in self.target_cols:
            nan_mask = self.future_predictions[col].isna()
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN values in {col} predictions. Filling with seasonal means.")
                
                # Use vectorized operations for faster filling
                month_values = self.future_predictions.loc[nan_mask, 'month']
                month_means = [monthly_means[col][m] for m in month_values]
                self.future_predictions.loc[nan_mask, col] = month_means
        
        # Final consistency check: ensure drought hierarchy is maintained
        print("\nApplying final consistency constraints...")
        for i, row in self.future_predictions.iterrows():
            for j in range(1, len(self.target_cols)):
                current_col = self.target_cols[j]
                prev_col = self.target_cols[j-1]
                if row[current_col] > row[prev_col]:
                    self.future_predictions.loc[i, current_col] = self.future_predictions.loc[i, prev_col]
        
        # Save predictions to CSV
        self.future_predictions.to_csv(f'{self.result_dir}/{self.abrv}/{self.abrv}_future_drought_predictions.csv', index=False)
        print(f"Predictions saved to '{self.result_dir}/{self.abrv}/{self.abrv}_future_drought_predictions.csv'")
        
        return self
    
    def create_visualizations(self):
        """
        Create visualizations of results.
        
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        self.print_section("VISUALIZATION AND ANALYSIS")
        
        if self.future_predictions is None:
            print("No predictions to visualize. Run make_predictions() first.")
            return self
        
        # Create enhanced visualizations
        plt.figure(figsize=(20, 15))
        
        # Set a modern aesthetic style
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        
        cmap = plt.cm.get_cmap('viridis')
        colors = [cmap(i/len(self.target_cols)) for i in range(len(self.target_cols))]
        
        for i, (col, color) in enumerate(zip(self.target_cols, colors)):
            plt.subplot(3, 2, i+1)
            
            # Plot historical data
            plt.plot(self.historical_df['date'], self.historical_df[col], 
                    color=color, alpha=0.7, linewidth=2, label='Historical')
            
            # Plot predictions
            plt.plot(self.future_predictions['date'], self.future_predictions[col], 
                    color='red', alpha=0.7, linewidth=2, linestyle='-', label='Predicted')
            
            # Add a vertical line at the transition point
            transition_date = self.historical_df['date'].max()
            plt.axvline(x=transition_date, color='black', linestyle='--', alpha=0.7,
                      label=f'Prediction Start ({transition_date.strftime("%Y-%m")})')
            
            # Add average CV R² to the plot
            r2_value = self.avg_results['r2'][col]
            rmse_value = self.avg_results['rmse'][col]
            
            plt.title(f'Drought Severity Index: {col}', fontsize=14, fontweight='bold')
            plt.ylabel('Percent Area (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add annotations
            textstr = f'R² = {r2_value:.4f}\nRMSE = {rmse_value:.4f}'
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, verticalalignment='top', bbox=props)
            
            # Customize date axis
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/drought_predictions.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved to '{self.result_dir}/drought_predictions.png'")
        
        # D4 Binary Classification Visualization
        self.print_section("D4 BINARY CLASSIFICATION ANALYSIS")
        
        # Ensure we have binary predictions and probabilities
        if 'D4_binary_pred' not in self.future_predictions.columns:
            print("Warning: D4 binary predictions not found in results. Generating from regression values...")
            self.future_predictions['D4_binary_pred'] = (self.future_predictions['D4'] > 0).astype(int)
            self.future_predictions['D4_binary_prob'] = np.minimum(1.0, self.future_predictions['D4'] / 5.0)
        
        # Count predicted D4 drought events
        d4_event_count = self.future_predictions['D4_binary_pred'].sum()
        d4_event_percentage = d4_event_count / len(self.future_predictions) * 100
        print(f"Predicted D4 drought events in future data: {d4_event_count} ({d4_event_percentage:.1f}%)")
        
        # Create detailed D4 binary classification visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Historical D4 values with binary threshold
        plt.subplot(3, 1, 1)
        plt.plot(self.historical_df['date'], self.historical_df['D4'], 
                color='purple', alpha=0.7, linewidth=2, label='Historical D4')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Binary Threshold (D4 > 0)')
        plt.title('Historical D4 Drought Index', fontsize=14, fontweight='bold')
        plt.ylabel('Percent Area (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: D4 future predictions with binary prediction
        plt.subplot(3, 1, 2)
        plt.plot(self.future_predictions['date'], self.future_predictions['D4'], 
                color='red', alpha=0.7, linewidth=2, label='D4 Regression')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        
        # Add a twin axis for probability
        ax2 = plt.twinx()
        ax2.plot(self.future_predictions['date'], self.future_predictions['D4_binary_prob'], 
                color='blue', alpha=0.7, linewidth=2, linestyle='--', label='D4 Probability')
        ax2.set_ylabel('Probability', color='blue', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim([0, 1])
        ax2.axhline(y=0.5, color='blue', linestyle='--', alpha=0.3, label='Decision Threshold (0.5)')
        
        plt.title('D4 Predictions: Regression vs. Probability', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Percent Area (%)')
        plt.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 3: Binary predictions (0/1)
        plt.subplot(3, 1, 3)
        plt.step(self.future_predictions['date'], self.future_predictions['D4_binary_pred'], 
                color='red', alpha=0.9, linewidth=2, where='mid')
        plt.fill_between(self.future_predictions['date'], self.future_predictions['D4_binary_pred'], 
                        step="mid", alpha=0.3, color='red')
        plt.title('Binary D4 Drought Prediction (0 = No Drought, 1 = Drought)', fontsize=14, fontweight='bold')
        plt.ylabel('Class', fontsize=12)
        plt.yticks([0, 1], ['No D4', 'D4 Drought'])
        plt.grid(True, alpha=0.3)
        
        # Add binary classification metrics
        textstr = (f'CV Accuracy = {self.avg_d4_binary["accuracy"]:.4f}\n'
                  f'CV F1 Score = {self.avg_d4_binary["f1"]:.4f}\n'
                  f'CV Precision = {self.avg_d4_binary["precision"]:.4f}\n'
                  f'CV Recall = {self.avg_d4_binary["recall"]:.4f}')
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/{self.abrv}/{self.abrv}_d4_binary_classification.png', dpi=300, bbox_inches='tight')
        print(f"D4 binary classification visualization saved to '{self.result_dir}/{self.abrv}/{self.abrv}_d4_binary_classification.png'")
        
        # Save final predictions with all binary data
        self.future_predictions.to_csv(f'{self.result_dir}/{self.abrv}/{self.abrv}_future_drought_predictions_with_binary.csv', index=False)
        print(f"Final predictions with binary classification saved to '{self.result_dir}/{self.abrv}/{self.abrv}_future_drought_predictions_with_binary.csv'")
        
        return self
    
    def create_uncertainty_visualization(self):
        """
        Create uncertainty visualizations using a robust bootstrap method.
        
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        self.print_section("UNCERTAINTY VISUALIZATION")
        
        if self.future_predictions is None:
            print("No predictions to visualize. Run make_predictions() first.")
            return self
        
        # Generate uncertainty visualization with a more robust approach
        plt.figure(figsize=(20, 15))
        
        cmap = plt.cm.get_cmap('viridis')
        colors = [cmap(i/len(self.target_cols)) for i in range(len(self.target_cols))]
        
        for i, (col, color) in enumerate(zip(self.target_cols, colors)):
            plt.subplot(3, 2, i+1)
            
            # Use bootstrap method for all models (more reliable)
            print(f"Generating uncertainty bands for {col} using bootstrap method...")
            
            # Convert monthly_stds to dictionary for vectorized mapping
            month_std_dict = self.monthly_stds[col].to_dict()
            
            # Apply function to get standard deviations for each month
            std_values = self.future_predictions['month'].apply(
                lambda m: month_std_dict.get(m, 0.1)
            ).values
            
            # Generate bootstrap samples
            n_bootstrap = 1000
            bootstrap_samples = np.random.normal(
                loc=self.future_predictions[col].values, 
                scale=std_values * 1.2,  # Scale up std slightly for better band width
                size=(n_bootstrap, len(self.future_predictions))
            )
            
            # Calculate quantiles from bootstrap samples
            lower_quantile = np.percentile(bootstrap_samples, 5, axis=0)
            upper_quantile = np.percentile(bootstrap_samples, 95, axis=0)
            
            # Plot historical data
            plt.plot(self.historical_df['date'], self.historical_df[col], 
                    color=color, alpha=0.7, linewidth=2, label='Historical')
            
            # Plot prediction mean
            plt.plot(self.future_predictions['date'], self.future_predictions[col], 
                    color='red', alpha=0.9, linewidth=2, label='Prediction')
            
            # Plot uncertainty band
            plt.fill_between(self.future_predictions['date'], lower_quantile, upper_quantile, 
                            color='red', alpha=0.2, label='90% Confidence Interval')
            
            # Add a vertical line at the transition point
            transition_date = self.historical_df['date'].max()
            plt.axvline(x=transition_date, color='black', linestyle='--', alpha=0.7,
                    label=f'Prediction Start ({transition_date.strftime("%Y-%m")})')
            
            plt.title(f'Drought Severity Index: {col} with Uncertainty', fontsize=14, fontweight='bold')
            plt.ylabel('Percent Area (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Customize date axis
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/{self.abrv}/{self.abrv}_drought_predictions_uncertainty.png', dpi=300, bbox_inches='tight')
        print(f"Uncertainty visualization saved to '{self.result_dir}/{self.abrv}/{self.abrv}_drought_predictions_uncertainty.png'")
        
        return self
    
    def generate_report(self):
        """
        Generate a comprehensive report of model performance and predictions.
        
        Returns:
        str
            Text report
        """
        self.print_section("PERFORMANCE")
        
        report = """
MODEL PERFORMANCE SUMMARY:
"""
        
        # Print summary metrics for each target
        for col in self.target_cols:
            report += f"{col} Regression Performance:\n"
            report += f"  R² Score: {self.avg_results['r2'][col]:.4f}\n"
            report += f"  MAE: {self.avg_results['mae'][col]:.4f}\n"
            report += f"  RMSE: {self.avg_results['rmse'][col]:.4f}\n\n"
        
        report += "D4 Binary Classification Performance:\n"
        report += f"  Accuracy: {self.avg_d4_binary['accuracy']:.4f}\n"
        report += f"  Precision: {self.avg_d4_binary['precision']:.4f}\n"
        report += f"  Recall: {self.avg_d4_binary['recall']:.4f}\n"
        report += f"  F1 Score: {self.avg_d4_binary['f1']:.4f}\n\n"
        
        # Calculate execution time
        end_time = time.time()
        total_time = end_time - self.start_time
        report += f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        
        print(report)
        
        # Save report to file
        with open(f'{self.result_dir}/{self.abrv}/{self.abrv}_model_report.txt', 'w') as f:
            f.write(report)
        print(f"Report saved to '{self.result_dir}/{self.abrv}/{self.abrv}_model_report.txt'")
        
        return report
    
    def run_complete_pipeline(self, historical_file, future_file):
        """
        Run the complete drought prediction pipeline from data loading to report generation.
        
        Parameters:
        historical_file : str
            Path to historical data CSV
        future_file : str
            Path to future data CSV
            
        Returns:
        self : DroughtPredictor
            Returns self for method chaining
        """
        return (self
                .load_data(historical_file, future_file)
                .prepare_features()
                .select_features()
                .cross_validate()
                .train_final_models()
                .make_predictions()
                .create_visualizations()
                .create_uncertainty_visualization()
                .generate_report()
                )