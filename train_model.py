import os
import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import joblib
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
# Assuming your client setup is reusable or you use python-binance directly
# If using BinanceFuturesClient, you might need to adapt data fetching slightly
from binance.client import Client 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"), # Log training process
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables (if needed for API keys)
load_dotenv()

# --- Constants ---
MODEL_FILENAME = 'proyecto_model.joblib'
SYMBOL = 'BTCUSDT'
INTERVAL = Client.KLINE_INTERVAL_15MINUTE # Use interval constants from the library
KLINE_START_DATE = "1 Jan, 2022" # How far back to fetch data (adjust as needed)
KLINE_END_DATE = "1 May, 2025"   # Up to when (adjust as needed)

# Feature parameters
RSI_PERIOD = 14
SMA_SHORT_PERIOD = 20
SMA_LONG_PERIOD = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14 # Added ATR period
LAG_PERIODS = [1, 2, 3]

# Target variable definition - Further adjustments
TARGET_LOOKAHEAD = 10 # Increased lookahead further
TARGET_THRESHOLD = 0.0020 # Increased threshold further (0.2%)

# --- Functions ---

def fetch_binance_data(client, symbol, interval, start_str, end_str):
    """Fetch historical klines from Binance."""
    logger.info(f"Fetching Binance klines for {symbol} ({interval}) from {start_str} to {end_str}...")
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to numeric, handle potential errors
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Convert timestamps
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['date'] = df['open_time'].dt.date # Add date column for merging F&G
        
        df = df[['date', 'open_time', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('open_time', inplace=True)
        logger.info(f"Fetched {len(df)} klines.")
        return df
    except Exception as e:
        logger.error(f"Error fetching Binance data: {e}", exc_info=True)
        return None

def fetch_fng_data(limit=0):
    """Fetch historical Fear & Greed index data."""
    logger.info(f"Fetching Fear & Greed data (limit={limit})...")
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json&date_format=cn"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d').dt.date
        df = df[['date', 'value']].rename(columns={'value': 'fear_and_greed'})
        df.drop_duplicates(subset=['date'], keep='last', inplace=True)
        df.set_index('date', inplace=True)
        logger.info(f"Fetched {len(df)} F&G records.")
        return df
    except Exception as e:
        logger.error(f"Error fetching F&G data: {e}", exc_info=True)
        return None

def calculate_and_lag_features(df):
    """Calculate technical indicators and create lagged features."""
    logger.info("Calculating features (RSI, SMAs, MACD, ATR, Volume) and lags...")
    if df is None or df.empty:
        return None
        
    # Ensure necessary columns are numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    
    # --- Calculate Base Indicators ---
    df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)
    df['sma_short'] = ta.sma(df['close'], length=SMA_SHORT_PERIOD)
    df['sma_long'] = ta.sma(df['close'], length=SMA_LONG_PERIOD)
    df['sma_diff'] = df['sma_short'] - df['sma_long']
    
    # Calculate MACD
    macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None and not macd.empty:
        df['macd_line'] = macd[f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df['macd_hist'] = macd[f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df['macd_signal'] = macd[f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
    else:
        logger.warning("Could not calculate MACD.")
        df['macd_line'] = np.nan
        df['macd_hist'] = np.nan
        df['macd_signal'] = np.nan

    # Calculate ATR (requires high, low, close)
    atr = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    if atr is not None and not atr.empty:
         df['atr'] = atr
    else:
         logger.warning("Could not calculate ATR.")
         df['atr'] = np.nan

    # Calculate Volume Change
    df['volume_change'] = df['volume'].pct_change()
    # Replace potential inf/-inf values resulting from pct_change if volume was 0
    df['volume_change'] = df['volume_change'].replace([np.inf, -np.inf], 0)
    # Fill initial NaN from pct_change
    df['volume_change'].fillna(0, inplace=True)

    # --- Create Lagged Features ---
    # Added 'atr' to features to lag
    features_to_lag = ['rsi', 'sma_diff', 'macd_hist', 'atr', 'volume_change']
    logger.info(f"Creating lags for: {features_to_lag}")
    for feature in features_to_lag:
        for lag in LAG_PERIODS:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
            
    return df

def create_target(df, lookahead=1, threshold=0.0):
    """Create the target variable based on future price movement."""
    logger.info(f"Creating target variable (lookahead={lookahead}, threshold={threshold * 100}%)...")
    if df is None or df.empty:
        return None
        
    # Calculate future return
    df['future_return'] = df['close'].pct_change(periods=-lookahead).shift(-lookahead)
    
    # Define target: 1 (Buy), -1 (Sell), 0 (Hold)
    df['target'] = 0
    df.loc[df['future_return'] > threshold, 'target'] = 1
    df.loc[df['future_return'] < -threshold, 'target'] = -1
    
    logger.info(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
    return df

def main():
    """Main training function."""
    logger.info("--- Starting Model Training (with enhanced features) ---")
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Initialize Binance Client (using public endpoints for historical data)
    # You might need API keys if your limits are too low without them
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    # Add check if keys are needed/present
    if not api_key or not api_secret:
        logger.warning("Binance API keys not found in .env. Using public client (may have rate limits).")
        client = Client() # Public client
    else:
        client = Client(api_key, api_secret)

    # 1. Fetch Data
    df_klines = fetch_binance_data(client, SYMBOL, INTERVAL, KLINE_START_DATE, KLINE_END_DATE)
    df_fng = fetch_fng_data(limit=0) # Fetch all available F&G

    if df_klines is None or df_fng is None:
        logger.error("Failed to fetch initial data. Exiting.")
        return

    # 2. Calculate Price Features and Lags
    df_klines = calculate_and_lag_features(df_klines)
    if df_klines is None:
        logger.error("Feature calculation failed. Exiting.")
        return

    # 3. Combine Data
    # Merge F&G data into klines data based on the date
    logger.info("Merging Kline features and F&G data...")
    df_klines.reset_index(inplace=True) # Need open_time as a column to merge date
    df_merged = pd.merge(df_klines, df_fng, on='date', how='left')
    df_merged['fear_and_greed'] = df_merged['fear_and_greed'].ffill()
    df_merged.set_index('open_time', inplace=True) # Set index back to open_time
    df_merged.sort_index(inplace=True) # Ensure data is sorted by time after merge

    # 4. Create Target Variable
    df_merged = create_target(df_merged, lookahead=TARGET_LOOKAHEAD, threshold=TARGET_THRESHOLD)

    # 5. Prepare Data for Model
    logger.info("Preparing data for training...")
    # Define the list of ALL features to be used by the model
    base_features = ['rsi', 'sma_diff', 'macd_line', 'macd_hist', 'macd_signal', 'atr', 'volume_change', 'fear_and_greed']
    lagged_feature_names = []
    features_to_lag = ['rsi', 'sma_diff', 'macd_hist', 'atr', 'volume_change']
    for feature in features_to_lag:
        for lag in LAG_PERIODS:
            lagged_feature_names.append(f'{feature}_lag_{lag}')
            
    features = base_features + lagged_feature_names
    logger.info(f"Using {len(features)} features: {features}")

    df_final = df_merged[features + ['target']].copy()
    
    # Drop rows with NaN values (more NaNs expected due to lagging)
    initial_rows = len(df_final)
    df_final.dropna(inplace=True)
    dropped_rows = initial_rows - len(df_final)
    logger.info(f"Dropped {dropped_rows} rows with NaN values (includes initial indicator/lag periods).")

    if df_final.empty:
        logger.error("No data left after cleaning NaN values. Check data fetching, feature calculation, and lag periods.")
        return
        
    X = df_final[features]
    y = df_final['target']

    # Check if target classes are imbalanced
    target_counts = y.value_counts()
    logger.info(f"Target class distribution before split:\n{target_counts}")
    if len(target_counts) < 3:
         logger.warning(f"Target variable has only {len(target_counts)} classes after processing. Check target definition or data range.")
         # Decide how to handle - maybe exit, maybe proceed with fewer classes?
         # For now, continue but be aware
         target_names = [str(c) for c in sorted(target_counts.index)]
    else:
         target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- ADD SCALING ---
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use transform only on test set
    # Save the scaler for potential use during live prediction
    scaler_filename = 'feature_scaler.joblib'
    joblib.dump(scaler, scaler_filename)
    logger.info(f"Scaler saved to {scaler_filename}")
    # ---------------------

    # 6. Hyperparameter Tuning with RandomizedSearchCV
    logger.info("Performing Hyperparameter Tuning with RandomizedSearchCV...")
    
    # Define the parameter distribution to search
    param_dist = {
        'n_estimators': randint(100, 500), # Number of trees
        'max_depth': [None, 10, 20, 30, 40, 50], # Max depth of trees
        'min_samples_split': randint(2, 20), # Min samples required to split a node
        'min_samples_leaf': randint(1, 20), # Min samples required at each leaf node
        'max_features': ['sqrt', 'log2', None] # Number of features to consider at each split
    }

    # Base model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

    # Randomized search setup (n_iter controls number of combinations tried)
    # cv is the number of cross-validation folds
    random_search = RandomizedSearchCV(estimator=rf, 
                                       param_distributions=param_dist, 
                                       n_iter=50, # Number of parameter settings sampled (adjust based on time)
                                       cv=3, # 3-fold cross-validation (adjust based on time/data size)
                                       verbose=1, # Show progress
                                       random_state=42, 
                                       n_jobs=-1, # Use all available cores
                                       scoring='accuracy') # Optimize for accuracy

    # Fit RandomizedSearchCV on the scaled training data
    random_search.fit(X_train_scaled, y_train)

    logger.info(f"Best parameters found by RandomizedSearchCV: {random_search.best_params_}")
    logger.info(f"Best cross-validation accuracy score: {random_search.best_score_:.4f}")

    # Use the best estimator found by the search
    best_model = random_search.best_estimator_
    
    # (Optional: Can re-train the best model on the full training set, 
    # but best_estimator_ is already trained on the full training set after CV)
    # logger.info("Training final model with best parameters...")
    # best_model.fit(X_train_scaled, y_train) 
    logger.info("Using best model found by RandomizedSearchCV.")

    # 7. Evaluate Final Model on Test Set
    logger.info("Evaluating best model on the (scaled) test set...")
    y_pred = best_model.predict(X_test_scaled) # Predict on scaled test data
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred)
    except ValueError as e:
         logger.error(f"Error generating classification report/matrix (likely due to missing classes in y_pred): {e}")
         report = "Error"
         matrix = "Error"
    
    logger.info(f"Test Set Accuracy (Tuned Model): {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{matrix}")

    # Feature Importance (from the best model)
    try:
        importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        logger.info(f"Feature Importances (Tuned Model - Top 15):\n{importances.head(15)}")
    except Exception as e:
        logger.warning(f"Could not calculate feature importances: {e}")

    # 8. Save Model and Scaler
    if accuracy > 0.52: 
        logger.info(f"Saving trained model ({MODEL_FILENAME}) and scaler ({scaler_filename})...")
        joblib.dump(best_model, MODEL_FILENAME) # Save the best model
        # Scaler was already saved earlier after fitting
        logger.info("Model and scaler saved successfully.")
    else:
        logger.warning(f"Tuned model accuracy ({accuracy:.4f}) is below threshold (0.52). Model not saved.")
        # Consider whether to keep the scaler even if model isn't saved

    logger.info("--- Training Process Finished ---")

if __name__ == "__main__":
    main() 