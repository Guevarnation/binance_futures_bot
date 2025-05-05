import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime
import requests
import pandas_ta as ta
from abc import ABC, abstractmethod
from binance_client import BinanceFuturesClient  # Assuming this import is correct
import joblib # For loading ML model
import os # For model path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Strategy:
    """Base strategy class"""
    def __init__(self, client):
        self.client = client
        self.symbol = client.symbol if hasattr(client, 'symbol') else 'BTCUSDT'
        self.interval = '15m'
        logger.info(f"Initialized base strategy for {self.symbol}")
    
    def execute(self):
        """Execute the strategy - to be implemented by subclasses"""
        raise NotImplementedError("Subclass must implement abstract method")


class SimpleMovingAverageStrategy(Strategy):
    """Simple Moving Average crossover strategy"""
    def __init__(self, client, short_window=20, long_window=50, interval='15m'):
        super().__init__(client)
        self.short_window = short_window
        self.long_window = long_window
        self.interval = interval
        logger.info(f"Initialized SMA strategy with short_window={short_window}, long_window={long_window}, interval={interval}")
    
    def get_historical_klines(self):
        """Get historical klines/candlesticks"""
        try:
            # Get enough klines to calculate the indicators
            klines = self.client.client.futures_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=self.long_window + 10  # Get a few extra to ensure we have enough data
            )
            
            # Create dataframe
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert price columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                              'quote_asset_volume', 'taker_buy_base_asset_volume',
                              'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            return df
        except Exception as e:
            logger.error(f"Failed to get historical klines: {e}")
            return None
    
    def calculate_signals(self, df):
        """Calculate trading signals based on SMA crossover"""
        if df is None or len(df) < self.long_window:
            logger.error("Not enough data to calculate signals")
            return None
        
        # Calculate SMAs
        df['short_sma'] = df['close'].rolling(window=self.short_window, min_periods=1).mean()
        df['long_sma'] = df['close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Calculate signals
        df['signal'] = 0.0
        df['signal'][self.short_window:] = np.where(
            df['short_sma'][self.short_window:] > df['long_sma'][self.short_window:], 1.0, 0.0
        )
        
        # Calculate position changes
        df['position'] = df['signal'].diff()
        
        logger.info(f"Calculated signals: last position = {df['position'].iloc[-1]}")
        return df
    
    def execute(self):
        """Execute the SMA crossover strategy"""
        logger.info("Executing SMA crossover strategy")
        
        # Get historical klines
        df = self.get_historical_klines()
        if df is None:
            return False
        
        # Calculate signals
        df = self.calculate_signals(df)
        if df is None:
            return False
        
        # Get current position
        positions = self.client.get_open_positions()
        current_position_amt = 0
        for pos in positions:
            if pos['symbol'] == self.symbol:
                current_position_amt = float(pos['positionAmt'])
                break
        
        # Check for trading signals
        last_position = df['position'].iloc[-1]
        current_price = float(df['close'].iloc[-1])
        
        if last_position == 1.0:  # Buy signal
            if current_position_amt <= 0:
                logger.info(f"BUY signal at {current_price}")
                
                # Close any existing short positions
                if current_position_amt < 0:
                    self.client.close_position(self.symbol)
                
                # Place buy order
                order = self.client.place_market_buy()
                if order:
                    # Place take profit and stop loss orders
                    self.client.place_take_profit_order(current_price, 'BUY')
                    self.client.place_stop_loss_order(current_price, 'BUY')
                    return True
        
        elif last_position == -1.0:  # Sell signal
            if current_position_amt >= 0:
                logger.info(f"SELL signal at {current_price}")
                
                # Close any existing long positions
                if current_position_amt > 0:
                    self.client.close_position(self.symbol)
                
                # Place sell order
                order = self.client.place_market_sell()
                if order:
                    # Place take profit and stop loss orders
                    self.client.place_take_profit_order(current_price, 'SELL')
                    self.client.place_stop_loss_order(current_price, 'SELL')
                    return True
        
        logger.info("No trading signal")
        return False


class RSIStrategy(Strategy):
    """RSI-based trading strategy"""
    def __init__(self, client, period=14, overbought=70, oversold=30, interval='15m'):
        super().__init__(client)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.interval = interval
        self.symbol = client.symbol
        logger.info(f"Initialized RSI strategy for {self.symbol} with period={period}, overbought={overbought}, oversold={oversold}, interval={interval}")
    
    def get_historical_klines(self):
        """Get historical klines/candlesticks"""
        try:
            # Get enough klines to calculate the indicators
            klines = self.client.client.futures_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=self.period + 50  # Get extra to ensure we have enough data
            )
            
            # Create dataframe
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert price columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                              'quote_asset_volume', 'taker_buy_base_asset_volume',
                              'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            return df
        except Exception as e:
            logger.error(f"Failed to get historical klines: {e}")
            return None
    
    def calculate_rsi(self, df):
        """Calculate RSI"""
        if df is None or len(df) < self.period + 1:
            logger.error("Not enough data to calculate RSI")
            return None
        
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        
        # Calculate gains and losses
        df['gain'] = df['price_change'].apply(lambda x: max(x, 0))
        df['loss'] = df['price_change'].apply(lambda x: max(-x, 0))
        
        # Calculate average gains and losses
        avg_gain = df['gain'].rolling(window=self.period, min_periods=1).mean()
        avg_loss = df['loss'].rolling(window=self.period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df['rsi'] < self.oversold, 1, df['signal'])  # Buy signal
        df['signal'] = np.where(df['rsi'] > self.overbought, -1, df['signal'])  # Sell signal
        
        logger.info(f"Latest RSI: {df['rsi'].iloc[-1]:.2f}")
        return df
    
    def execute(self):
        """Execute RSI strategy"""
        logger.info("Executing RSI strategy")
        
        # Get historical klines
        df = self.get_historical_klines()
        if df is None:
            return False
        
        # Calculate RSI and signals
        df = self.calculate_rsi(df)
        if df is None:
            return False
        
        # Get current position
        positions = self.client.get_open_positions()
        current_position_amt = 0
        for pos in positions:
            if pos['symbol'] == self.symbol:
                current_position_amt = float(pos['positionAmt'])
                break
        
        # Check for trading signals
        last_signal = df['signal'].iloc[-1]
        current_price = float(df['close'].iloc[-1])
        current_rsi = df['rsi'].iloc[-1]
        
        if last_signal == 1 and current_position_amt <= 0:  # Buy signal
            logger.info(f"BUY signal at {current_price} (RSI: {current_rsi:.2f})")
            
            # Close any existing short positions
            if current_position_amt < 0:
                self.client.close_position(self.symbol)
            
            # Place buy order
            order = self.client.place_market_buy()
            if order:
                # Place take profit and stop loss orders
                self.client.place_take_profit_order(current_price, 'BUY')
                self.client.place_stop_loss_order(current_price, 'BUY')
                return True
                
        elif last_signal == -1 and current_position_amt >= 0:  # Sell signal
            logger.info(f"SELL signal at {current_price} (RSI: {current_rsi:.2f})")
            
            # Close any existing long positions
            if current_position_amt > 0:
                self.client.close_position(self.symbol)
            
            # Place sell order
            order = self.client.place_market_sell()
            if order:
                # Place take profit and stop loss orders
                self.client.place_take_profit_order(current_price, 'SELL')
                self.client.place_stop_loss_order(current_price, 'SELL')
                return True
        
        logger.info(f"No trading signal (RSI: {current_rsi:.2f})")
        return False 

class ProyectoStrategy(Strategy):
    """
    Machine Learning based trading strategy using a Random Forest model.
    Loads a pre-trained model ('proyecto_model.joblib' by default) to make decisions.
    Features used (example): RSI, SMA_short, SMA_long, Fear & Greed Index.
    """
    def __init__(self, client: BinanceFuturesClient, symbol='BTCUSDT', interval='15m', model_path='proyecto_model.joblib', scaler_path='feature_scaler.joblib'):
        super().__init__(client)
        self.symbol = symbol
        self.interval = interval
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.fng_url = "https://api.alternative.me/fng/"
        
        # Define the features the loaded Random Forest model expects
        # MUST match the features used during training in train_model.py!
        self.base_features = ['rsi', 'sma_diff', 'macd_line', 'macd_hist', 'macd_signal', 'atr', 'volume_change', 'fear_and_greed']
        self.lag_periods = [1, 2, 3] # Must match train_model.py
        self.features_to_lag = ['rsi', 'sma_diff', 'macd_hist', 'atr', 'volume_change'] # Must match train_model.py
        self.lagged_feature_names = []
        for feature in self.features_to_lag:
            for lag in self.lag_periods:
                self.lagged_feature_names.append(f'{feature}_lag_{lag}')
        self.required_features = self.base_features + self.lagged_feature_names

        # Store feature calculation parameters (must match train_model.py)
        self.feature_params = {
            'rsi_period': 14,
            'sma_short': 20,
            'sma_long': 50,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14 
        }
        logger.info(f"Initialized ProyectoStrategy (Random Forest) for {self.symbol} ({self.interval})")
        logger.info(f"Using model: {self.model_path}, Scaler: {self.scaler_path}")
        logger.info(f"Expecting {len(self.required_features)} features: {self.required_features}")
        if not self.model:
             logger.warning(f"Model not loaded from {self.model_path}. Strategy will not trade.")
        if not self.scaler:
             logger.warning(f"Scaler not loaded from {self.scaler_path}. Strategy cannot scale features and will not trade.")

    def _load_model(self):
        """Loads the pre-trained Random Forest model."""
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                # Basic check if it looks like a scikit-learn model (specifically RF)
                if hasattr(model, 'predict') and hasattr(model, 'estimators_'):
                     logger.info(f"Successfully loaded Random Forest model from {self.model_path}")
                     return model
                else:
                     logger.error(f"Loaded object from {self.model_path} does not appear to be a Random Forest model.")
                     return None
            except Exception as e:
                logger.error(f"Error loading model from {self.model_path}: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Model file not found at {self.model_path}. Please train and save the model first.")
            return None

    def _load_scaler(self):
        """Loads the pre-trained feature scaler."""
        if os.path.exists(self.scaler_path):
            try:
                scaler = joblib.load(self.scaler_path)
                logger.info(f"Successfully loaded scaler from {self.scaler_path}")
                return scaler
            except Exception as e:
                logger.error(f"Error loading scaler from {self.scaler_path}: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Scaler file not found at {self.scaler_path}. Please ensure training was successful and saved the scaler.")
            return None

    def _get_fear_and_greed_index(self):
        """Fetches the Fear and Greed Index."""
        try:
            response = requests.get(self.fng_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'data' in data and len(data['data']) > 0 and 'value' in data['data'][0]:
                 fng_value = int(data['data'][0]['value'])
                 logger.info(f"Current Fear and Greed Index: {fng_value}")
                 return fng_value
            else:
                 logger.error(f"Unexpected data format from Fear and Greed API: {data}")
                 return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not fetch Fear and Greed Index: {e}")
            return None
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing Fear and Greed Index data: {e}")
            return None

    def _get_features(self):
        """Fetches market data, calculates all required features, and scales them."""
        if not self.model or not self.scaler:
            logger.error("Model or Scaler not loaded, cannot get features.")
            return None

        # --- Fetch F&G ---
        fng_value = self._get_fear_and_greed_index()
        if fng_value is None:
            logger.warning("Could not fetch F&G index. Skipping feature generation.")
            return None

        # --- Fetch Kline Data ---
        # Determine required kline limit based on longest calculation needed (e.g., SMA_long + max lag)
        longest_sma = self.feature_params.get('sma_long', 50)
        longest_macd = self.feature_params.get('macd_slow', 26)
        longest_rsi = self.feature_params.get('rsi_period', 14)
        longest_atr = self.feature_params.get('atr_period', 14)
        max_lag = max(self.lag_periods) if self.lag_periods else 0
        # Need enough lookback for the longest indicator PLUS the max lag PLUS some buffer
        limit = max(longest_sma, longest_macd, longest_rsi, longest_atr) + max_lag + 50 
        logger.debug(f"Fetching {limit} klines for feature calculation...")
        
        try:
            klines = self.client.client.futures_klines(
                 symbol=self.symbol,
                 interval=self.interval,
                 limit=limit
            )
            if not klines or len(klines) < max(longest_sma, longest_macd, longest_rsi, longest_atr) + max_lag:
                 logger.warning(f"Insufficient kline data fetched for {self.interval} ({len(klines) if klines else 0}) needed for indicators + lags.")
                 return None

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            
            # Convert necessary columns to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True) # Drop rows if conversion failed
                 
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('open_time', inplace=True)

        except Exception as e:
            logger.error(f"Failed to get {self.interval} klines for features: {e}")
            return None

        # --- Calculate Technical Indicators (matching train_model.py) ---
        logger.debug("Calculating base technical indicators...")
        df['rsi'] = ta.rsi(df['close'], length=self.feature_params['rsi_period'])
        df['sma_short'] = ta.sma(df['close'], length=self.feature_params['sma_short'])
        df['sma_long'] = ta.sma(df['close'], length=self.feature_params['sma_long'])
        df['sma_diff'] = df['sma_short'] - df['sma_long']
        
        macd = ta.macd(df['close'], fast=self.feature_params['macd_fast'], slow=self.feature_params['macd_slow'], signal=self.feature_params['macd_signal'])
        if macd is not None and not macd.empty:
            df['macd_line'] = macd[f'MACD_{self.feature_params["macd_fast"]}_{self.feature_params["macd_slow"]}_{self.feature_params["macd_signal"]}']
            df['macd_hist'] = macd[f'MACDh_{self.feature_params["macd_fast"]}_{self.feature_params["macd_slow"]}_{self.feature_params["macd_signal"]}']
            df['macd_signal'] = macd[f'MACDs_{self.feature_params["macd_fast"]}_{self.feature_params["macd_slow"]}_{self.feature_params["macd_signal"]}']
        else:
             df['macd_line'], df['macd_hist'], df['macd_signal'] = np.nan, np.nan, np.nan

        atr = ta.atr(df['high'], df['low'], df['close'], length=self.feature_params['atr_period'])
        if atr is not None and not atr.empty:
            df['atr'] = atr
        else:
             df['atr'] = np.nan
             
        df['volume_change'] = df['volume'].pct_change()
        df['volume_change'] = df['volume_change'].replace([np.inf, -np.inf], 0)
        df['volume_change'].fillna(0, inplace=True)
        
        # --- Create Lagged Features ---
        logger.debug(f"Creating lags for: {self.features_to_lag}")
        for feature in self.features_to_lag:
            for lag in self.lag_periods:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

        # --- Add F&G ---
        df['fear_and_greed'] = fng_value # Assign current F&G value
        
        # --- Prepare Final Feature Vector for the latest timestamp ---
        latest_data = df.iloc[-1:].copy() # Get the last row as a DataFrame

        # Check for NaN values in the latest row for required features
        if latest_data[self.required_features].isnull().any().any():
             logger.warning(f"Latest features contain NaN values: {latest_data[self.required_features].isnull().sum()}")
             logger.debug(f"Latest data row with NaNs:\n{latest_data[self.required_features]}")
             return None

        # Ensure feature order and select only required features
        try:
            final_features_df = latest_data[self.required_features]
        except KeyError as e:
             logger.error(f"Feature mismatch error during live prediction: Model expects feature '{e}'. Check self.required_features list.")
             return None

        # --- Scale the features --- 
        logger.debug("Scaling latest features...")
        try:
            scaled_features = self.scaler.transform(final_features_df) # Use the loaded scaler
            logger.debug(f"Scaled feature vector shape: {scaled_features.shape}")
            logger.debug(f"Scaled feature vector: {scaled_features}")
            return scaled_features # Return the scaled numpy array
        except Exception as e:
            logger.error(f"Error scaling features: {e}", exc_info=True)
            return None
        
    def execute(self):
        """Executes the ML strategy: get features, predict, trade."""
        logger.info(f"Executing ProyectoStrategy (Random Forest) for {self.symbol} ({self.interval})...")
        if not self.model or not self.scaler:
            logger.error("Model or Scaler is not loaded. Cannot execute strategy.")
            return

        # Get current SCALED features
        scaled_features = self._get_features() # This now returns the scaled features
        if scaled_features is None:
            logger.warning("Could not get valid scaled features for prediction. Skipping cycle.")
            return

        # Predict using the loaded model and scaled features
        try:
            # Ensure input shape is correct for prediction (e.g., (1, n_features))
            if scaled_features.shape[0] == 1 and scaled_features.shape[1] == len(self.required_features):
                 prediction = self.model.predict(scaled_features)
                 signal = int(prediction[0])
                 logger.info(f"Random Forest Model prediction: {signal} (1=Buy, -1=Sell, 0=Hold)")
            else:
                 logger.error(f"Scaled feature vector has unexpected shape: {scaled_features.shape}. Expected (1, {len(self.required_features)}). Cannot predict.")
                 return
        except Exception as e:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
            return

        # Get current position status
        try:
            positions = self.client.client.futures_position_information(symbol=self.symbol)
            position_amount = 0.0
            pos_info = next((p for p in positions if p.get('symbol') == self.symbol), None)
            if pos_info:
                position_amount = float(pos_info.get('positionAmt', 0))
            logger.info(f"Current position amount for {self.symbol}: {position_amount}")
        except Exception as e:
            logger.error(f"Failed to get current position: {e}")
            return # Can't make trade decision without knowing position

        # --- Trade Execution Logic ---
        # Calculate quantity based on fixed notional value to meet minimums
        target_notional = 200 # Increased target notional to ensure rounded quantity meets minimum

        # Get current price for quantity calculation
        try:
            ticker_info = self.client.client.futures_mark_price(symbol=self.symbol)
            current_price = float(ticker_info['markPrice'])
            if current_price <= 0:
                 logger.error("Current mark price is zero or negative, cannot calculate quantity.")
                 return
            trade_quantity = round(target_notional / current_price, 3) # Round to appropriate precision for BTC (3 decimals usually ok)
            logger.info(f"Target notional: {target_notional} USDT, Current Price: {current_price}, Calculated Quantity: {trade_quantity}")

            # Add a check for minimum quantity if the exchange has one (optional but good practice)
            # min_qty = ... get from exchange info ...
            # if trade_quantity < min_qty:
            #    logger.warning(f"Calculated quantity {trade_quantity} is below minimum required {min_qty}. Adjust target_notional.")
            #    trade_quantity = min_qty

        except Exception as e:
             logger.error(f"Failed to get current price or calculate quantity: {e}")
             return

        if signal == 1: # Buy Signal
            logger.info(f"BUY signal predicted by Random Forest model.")
            if position_amount <= 0: # Only buy if not already long
                if position_amount < 0: # Close existing short position first
                    logger.info(f"Closing existing short position ({position_amount}) before buying.")
                    self.client.close_position(self.symbol)
                    time.sleep(1) # Allow time for closure order to process
                logger.info(f"Placing MARKET BUY order for {trade_quantity} {self.symbol} (Notional: ~{target_notional} USDT) based on RF signal.")
                order = self.client.place_market_buy(quantity=trade_quantity)
                if order:
                     logger.info(f"BUY order placed successfully: {order}")
                     # Optional: Place TP/SL based on prediction confidence or fixed rules
                else:
                     logger.error("Failed to place BUY order.")
            else:
                logger.info("Already in a long position. No action taken on BUY signal.")

        elif signal == -1: # Sell Signal
            logger.info(f"SELL signal predicted by Random Forest model.")
            if position_amount >= 0: # Only sell if not already short
                if position_amount > 0: # Close existing long position first
                    logger.info(f"Closing existing long position ({position_amount}) before selling.")
                    self.client.close_position(self.symbol)
                    time.sleep(1) # Allow time for closure order to process
                logger.info(f"Placing MARKET SELL order for {trade_quantity} {self.symbol} (Notional: ~{target_notional} USDT) based on RF signal.")
                order = self.client.place_market_sell(quantity=trade_quantity)
                if order:
                     logger.info(f"SELL order placed successfully: {order}")
                     # Optional: Place TP/SL based on prediction confidence or fixed rules
                else:
                     logger.error("Failed to place SELL order.")
            else:
                logger.info("Already in a short position. No action taken on SELL signal.")

        elif signal == 0: # Hold Signal
             logger.info("HOLD signal predicted by model. No action taken.")

        else:
             logger.warning(f"Received unknown signal from model: {signal}. Expected 1, -1, or 0.")

    def _calculate_and_lag_features_for_live(self, df):
        """Calculate features needed for prediction using the latest data."""
        # This is similar to train_model.py but adapted for live, single-point prediction
        # Needs enough historical data in df to calculate all indicators and lags
        logger.debug("Calculating live features...")
        if df is None or df.empty or len(df) < self.min_data_points:
            logger.warning(f"Not enough live data points ({len(df) if df is not None else 0}) to calculate features. Need at least {self.min_data_points}.")
            return None

        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 1. Base Indicators
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        df['sma_short'] = ta.sma(df['close'], length=self.sma_short_period)
        df['sma_long'] = ta.sma(df['close'], length=self.sma_long_period)
        df['sma_diff'] = df['sma_short'] - df['sma_long']
        macd = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd is not None and not macd.empty:
            df['macd_line'] = macd[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            df['macd_hist'] = macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            df['macd_signal'] = macd[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
        else:
            df['macd_line'] = np.nan
            df['macd_hist'] = np.nan
            df['macd_signal'] = np.nan
            
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        df['volume_change'] = df['volume'].pct_change() * 100
        df['volume_change'].fillna(0, inplace=True) # Fill first NaN

        # Fetch F&G - Assume it applies to the *current* state when predicting
        fng_value = self._fetch_fng_index()
        if fng_value is None:
            logger.warning("Could not fetch F&G index, using NaN.")
            fng_value = np.nan # Or use last known value / default?
        df['fear_and_greed'] = fng_value # Apply to the whole series for simplicity, will use latest row

        # 2. Lagged Features
        for feature in self.features_to_lag:
            for lag in self.lag_periods:
                 df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

        # 3. Select and Return Latest Features
        latest_features = df.iloc[-1][self.required_feature_order]

        # Check for NaNs in the final feature set
        if latest_features.isnull().any():
            logger.warning(f"NaN values found in latest feature set: \n{latest_features[latest_features.isnull()]}")
            # Optional: Impute NaNs if appropriate, e.g., with mean, median, or forward fill
            # For now, returning None to prevent prediction with incomplete data
            return None
            
        return latest_features.values.reshape(1, -1) # Return as 2D array for scaler/model

class MLStrategy(Strategy):
    """
    Machine Learning based trading strategy.
    Loads a pre-trained model to make decisions.
    """
    def __init__(self, client: BinanceFuturesClient, symbol='BTCUSDT', interval='15m', model_path='ml_model.joblib'):
        super().__init__(client)
        self.symbol = symbol
        self.interval = interval
        self.model_path = model_path
        self.model = self._load_model()
        self.fng_url = "https://api.alternative.me/fng/"
        # Define the features the loaded model expects (example - needs to match training)
        self.required_features = ['rsi', 'sma_short', 'sma_long', 'fear_and_greed'] # Example
        self.feature_params = {'rsi_period': 14, 'sma_short': 20, 'sma_long': 50} # Example
        logger.info(f"Initialized MLStrategy for {self.symbol} ({self.interval}) using model: {self.model_path}")

    def _load_model(self):
        """Loads the pre-trained ML model."""
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logger.info(f"Successfully loaded model from {self.model_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading model from {self.model_path}: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Model file not found at {self.model_path}. Please train and save the model first.")
            return None

    def _get_fear_and_greed_index(self):
        """Fetches the Fear and Greed Index."""
        try:
            response = requests.get(self.fng_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'data' in data and len(data['data']) > 0 and 'value' in data['data'][0]:
                 fng_value = int(data['data'][0]['value'])
                 logger.info(f"Current Fear and Greed Index: {fng_value}")
                 return fng_value
            else:
                 logger.error(f"Unexpected data format from Fear and Greed API: {data}")
                 return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not fetch Fear and Greed Index: {e}")
            return None
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing Fear and Greed Index data: {e}")
            return None

    def _get_features(self):
        """Fetches market data and calculates features required by the model."""
        if not self.model:
            logger.error("Model not loaded, cannot get features.")
            return None

        # --- Fetch F&G --- 
        fng_value = self._get_fear_and_greed_index()
        if fng_value is None:
            return None # Cannot proceed without all features

        # --- Fetch Kline Data --- 
        # Determine required kline limit based on feature calculations needed
        # Example: need at least sma_long periods for SMA
        limit = self.feature_params.get('sma_long', 50) + 50 # Ensure enough data
        try:
            klines = self.client.client.futures_klines(
                 symbol=self.symbol,
                 interval=self.interval,
                 limit=limit
            )
            if not klines or len(klines) < limit - 50:
                 logger.warning(f"Insufficient kline data fetched ({len(klines) if klines else 0} < {limit-50})")
                 return None

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['close'] = pd.to_numeric(df['close'])

        except Exception as e:
            logger.error(f"Failed to get {self.interval} klines for features: {e}")
            return None

        # --- Calculate Technical Indicators --- 
        # Example: Calculate features defined in self.required_features
        features = pd.DataFrame(index=df.index)
        if 'rsi' in self.required_features:
             rsi_period = self.feature_params.get('rsi_period', 14)
             features['rsi'] = ta.rsi(df['close'], length=rsi_period)
        if 'sma_short' in self.required_features:
             sma_short_period = self.feature_params.get('sma_short', 20)
             features['sma_short'] = ta.sma(df['close'], length=sma_short_period)
        if 'sma_long' in self.required_features:
             sma_long_period = self.feature_params.get('sma_long', 50)
             features['sma_long'] = ta.sma(df['close'], length=sma_long_period)
        # Add more feature calculations here as needed...

        # --- Add F&G --- 
        if 'fear_and_greed' in self.required_features:
             features['fear_and_greed'] = fng_value
        
        # --- Prepare Final Feature Vector ---
        # Get the latest complete feature set
        latest_features = features.iloc[-1].copy()

        # Ensure all required features are present and not NaN
        if latest_features.isnull().any():
             logger.warning(f"Latest features contain NaN values: {latest_features[latest_features.isnull()]}")
             return None

        # Ensure the order matches the training order (important!)
        try:
            final_feature_vector = latest_features[self.required_features].values.reshape(1, -1)
            logger.debug(f"Latest feature vector: {final_feature_vector}")
            return final_feature_vector
        except KeyError as e:
             logger.error(f"Feature mismatch between required_features and calculated features: Missing {e}")
             return None
        
    def execute(self):
        """Executes the ML strategy: get features, predict, trade."""
        logger.info(f"Executing ML Strategy for {self.symbol} ({self.interval})...")
        if not self.model:
            logger.error("Model is not loaded. Cannot execute strategy.")
            return

        # Get current features
        features = self._get_features()
        if features is None:
            logger.warning("Could not get features for prediction. Skipping cycle.")
            return

        # Predict using the loaded model
        try:
            prediction = self.model.predict(features)
            signal = int(prediction[0]) # Assuming model outputs 1 (Buy), -1 (Sell), 0 (Hold)
            logger.info(f"Model prediction: {signal}")
        except Exception as e:
            logger.error(f"Error during model prediction: {e}", exc_info=True)
            return

        # Get current position status
        try:
            positions = self.client.client.futures_position_information(symbol=self.symbol)
            position_amount = 0.0
            pos_info = next((p for p in positions if p.get('symbol') == self.symbol), None)
            if pos_info:
                position_amount = float(pos_info.get('positionAmt', 0))
        except Exception as e:
            logger.error(f"Failed to get current position: {e}")
            return # Can't make trade decision without knowing position

        # --- Trade Execution Logic --- 
        # Calculate quantity based on fixed notional value to meet minimums
        target_notional = 200 # Increased target notional to ensure rounded quantity meets minimum
        
        # Get current price for quantity calculation
        try:
            ticker_info = self.client.client.futures_mark_price(symbol=self.symbol)
            current_price = float(ticker_info['markPrice'])
            if current_price <= 0:
                 logger.error("Current mark price is zero or negative, cannot calculate quantity.")
                 return
            trade_quantity = round(target_notional / current_price, 3) # Round to appropriate precision for BTC (3 decimals usually ok)
            logger.info(f"Target notional: {target_notional} USDT, Current Price: {current_price}, Calculated Quantity: {trade_quantity}")
            
            # Add a check for minimum quantity if the exchange has one (optional but good practice)
            # min_qty = ... get from exchange info ...
            # if trade_quantity < min_qty:
            #    logger.warning(f"Calculated quantity {trade_quantity} is below minimum required {min_qty}. Adjust target_notional.")
            #    trade_quantity = min_qty 

        except Exception as e:
             logger.error(f"Failed to get current price or calculate quantity: {e}")
             return

        if signal == 1: # Buy Signal
            logger.info(f"BUY signal predicted by model.")
            if position_amount <= 0: # Only buy if not already long
                if position_amount < 0: # Close existing short position first
                    logger.info(f"Closing existing short position ({position_amount}) before buying.")
                    self.client.close_position(self.symbol)
                    time.sleep(1) # Allow time for closure
                logger.info(f"Placing MARKET BUY order for {trade_quantity} {self.symbol} (Notional: ~{target_notional} USDT) based on ML signal.")
                order = self.client.place_market_buy(quantity=trade_quantity)
                if order:
                     logger.info(f"BUY order placed successfully: {order}")
                     # Optional: Place TP/SL based on prediction or fixed rules
                else:
                     logger.error("Failed to place BUY order.")
            else:
                logger.info("Already in a long position. No action taken on BUY signal.")

        elif signal == -1: # Sell Signal
            logger.info(f"SELL signal predicted by model.")
            if position_amount >= 0: # Only sell if not already short
                if position_amount > 0: # Close existing long position first
                    logger.info(f"Closing existing long position ({position_amount}) before selling.")
                    self.client.close_position(self.symbol)
                    time.sleep(1) # Allow time for closure
                logger.info(f"Placing MARKET SELL order for {trade_quantity} {self.symbol} (Notional: ~{target_notional} USDT) based on ML signal.")
                order = self.client.place_market_sell(quantity=trade_quantity)
                if order:
                     logger.info(f"SELL order placed successfully: {order}")
                     # Optional: Place TP/SL based on prediction or fixed rules
                else:
                     logger.error("Failed to place SELL order.")
            else:
                logger.info("Already in a short position. No action taken on SELL signal.")
        
        elif signal == 0: # Hold Signal
             logger.info("HOLD signal predicted by model. No action taken.")
        
        else:
             logger.warning(f"Received unknown signal from model: {signal}")

# Make sure this line exists at the end or is updated if already present
# This dictionary maps strategy names (used in CLI args) to classes
STRATEGY_MAP = {
    'sma': SimpleMovingAverageStrategy,
    'rsi': RSIStrategy,
    'proyecto': ProyectoStrategy, # Now points to the new RF-based strategy
    'ml': MLStrategy, # Keep the generic ML strategy if needed, or remove if redundant
}
# Ensure STRATEGY_MAP is defined or updated appropriately at the end of the file.

def get_strategy(strategy_name, client, interval):
    if strategy_name.lower() in STRATEGY_MAP:
        StrategyClass = STRATEGY_MAP[strategy_name.lower()]
        # Inspect the StrategyClass constructor to pass correct args
        # This part needs refinement based on how args are passed from main.py
        # Simple approach: Pass client and interval, use defaults for others
        # More complex: Pass all relevant args from main.py if needed
        try:
             # Check if interval is expected by the constructor
             import inspect
             sig = inspect.signature(StrategyClass.__init__)
             if 'interval' in sig.parameters:
                 return StrategyClass(client=client, interval=interval)
             else:
                 # If interval is not in __init__ (like base Strategy maybe?)
                 # Or handle strategies that don't need interval specifically
                 return StrategyClass(client=client)
        except Exception as e:
             logger.error(f"Error initializing strategy {strategy_name}: {e}")
             return None
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return None 