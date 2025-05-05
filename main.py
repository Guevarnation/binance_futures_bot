import os
import logging
import argparse
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from binance_client import BinanceFuturesClient
from strategies import SimpleMovingAverageStrategy, RSIStrategy, get_strategy
import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Results Logging Function ---
RESULTS_FILE = 'trade_results.csv'

def log_results_to_csv(start_time, end_time, strategy_name, symbol, interval, initial_balance, final_balance):
    """Appends trading session results to a CSV file."""
    pnl = final_balance - initial_balance
    pnl_percent = (pnl / initial_balance) * 100 if initial_balance else 0
    
    results_data = {
        'start_time': [start_time.strftime('%Y-%m-%d %H:%M:%S')],
        'end_time': [end_time.strftime('%Y-%m-%d %H:%M:%S')],
        'strategy': [strategy_name],
        'symbol': [symbol],
        'interval': [interval],
        'initial_usdt': [round(initial_balance, 2)],
        'final_usdt': [round(final_balance, 2)],
        'pnl_usdt': [round(pnl, 2)],
        'pnl_percent': [round(pnl_percent, 2)]
    }
    df_results = pd.DataFrame(results_data)
    
    try:
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(RESULTS_FILE)
        df_results.to_csv(RESULTS_FILE, mode='a', header=not file_exists, index=False)
        logger.info(f"Results appended to {RESULTS_FILE}")
    except Exception as e:
        logger.error(f"Failed to write results to {RESULTS_FILE}: {e}")

def run_bot(strategy_name, symbol, interval, once):
    """Main function to run the trading bot."""
    logger.info(f"Starting bot for {symbol} on {interval} interval with strategy '{strategy_name}'.")
    
    # Load environment variables from .env file
    load_dotenv() 
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    
    # --- Read TRADE_NOTIONAL from .env --- 
    trade_notional_str = os.getenv('TRADE_NOTIONAL', '200') # Default to '200' if not found
    try:
        trade_notional = float(trade_notional_str)
        logger.info(f"Using TRADE_NOTIONAL value: {trade_notional}")
    except ValueError:
        logger.error(f"Invalid TRADE_NOTIONAL value '{trade_notional_str}' in .env file. Using default 200.")
        trade_notional = 200.0
    # -------------------------------------

    if not api_key or not api_secret:
        logger.error("Binance API key or secret key not found in environment variables.")
        return

    # Initialize Binance Client
    client = BinanceFuturesClient(api_key, api_secret, symbol=symbol)
    if not client.test_connection():
        logger.error("Failed to connect to Binance API.")
        return
        
    # Fetch initial balance for P&L tracking
    initial_balance = client.get_account_usdt_balance()
    if initial_balance is None:
        logger.error("Could not fetch initial account balance. Exiting.")
        return
    logger.info(f"Initial Account USDT balance: {initial_balance:.2f}")
    start_time = datetime.now()

    # Create strategy instance using the factory function
    # Pass the trade_notional value to the factory
    strategy_kwargs = {}
    if strategy_name.lower() == 'proyecto':
        strategy_kwargs['trade_notional'] = trade_notional
        # Pass model/scaler paths if they become configurable
        # strategy_kwargs['model_path'] = os.getenv('MODEL_PATH', 'proyecto_model.joblib')
        # strategy_kwargs['scaler_path'] = os.getenv('SCALER_PATH', 'feature_scaler.joblib')

    strategy = get_strategy(strategy_name, client=client, interval=interval, **strategy_kwargs)

    if not strategy:
        logger.error(f"Could not create strategy '{strategy_name}'. Exiting.")
        return

    # Main loop
    try:
        while True:
            logger.info(f"--- Running strategy cycle ({datetime.now()}) ---")
            try:
                strategy.execute()
            except Exception as e:
                logger.error(f"Error during strategy execution: {e}", exc_info=True)
            
            if once:
                logger.info("Running in 'once' mode. Exiting after one cycle.")
                break

            # Wait for the next candle
            try:
                wait_seconds = utils.calculate_wait_time(interval)
                logger.info(f"Waiting {wait_seconds:.2f} seconds for next {interval} candle")
                time.sleep(wait_seconds)
            except ValueError as e:
                logger.error(f"Error calculating wait time: {e}. Waiting 60 seconds as fallback.")
                time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Bot stopped manually by user (KeyboardInterrupt).")
    finally:
        # Log results on exit (normal or interrupt)
        logger.info("--- Bot shutting down --- ")
        final_balance = client.get_account_usdt_balance()
        if final_balance is not None and initial_balance is not None:
             logger.info(f"Final Account USDT balance: {final_balance:.2f}")
             end_time = datetime.now()
             log_results_to_csv(start_time, end_time, strategy_name, symbol, interval, initial_balance, final_balance)
        else:
             logger.error("Could not fetch final balance for P&L calculation.")
        # Optional: Close any open positions on exit?
        # current_pos = client.get_position_amount(symbol)
        # if current_pos != 0:
        #     logger.info(f"Closing open position ({current_pos}) on exit...")
        #     side = 'BUY' if current_pos < 0 else 'SELL'
        #     qty = abs(current_pos)
        #     client.place_order(symbol, side, 'MARKET', quantity=qty)
        logger.info("Bot finished.")

def main():
    """Main entry point"""
    # Create log directory
    os.makedirs('logs', exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    parser.add_argument('--strategy', type=str, default='proyecto', choices=['sma', 'rsi', 'proyecto'],
                        help='Trading strategy to use')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                        help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, default='15m',
                        help='Trading interval (e.g., 1m, 5m, 15m, 1h)')
    parser.add_argument('--once', action='store_true', 
                        help='Run the strategy logic only once and exit.')

    args = parser.parse_args()

    run_bot(args.strategy, args.symbol, args.interval, args.once)

if __name__ == "__main__":
    main() 