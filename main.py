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

def run_bot(strategy_name, interval):
    """Main bot execution loop"""
    logger.info(f"Starting bot with strategy '{strategy_name}' on interval '{interval}'")

    # Initialize client
    client = BinanceFuturesClient()
    if not client.client:
        logger.error("Failed to initialize Binance client. Check API keys and connection.")
        return

    # Set symbol in client (can be overridden by strategy if needed)
    client.symbol = 'BTCUSDT' # Default or get from args

    # Verify connectivity and get initial balance
    try:
        initial_balance_info = client.get_account_balance()
        if initial_balance_info and 'USDT' in initial_balance_info:
            initial_balance = initial_balance_info['USDT']
            logger.info(f"Initial Account USDT balance: {initial_balance:.2f}")
        else:
            logger.error("Could not retrieve initial USDT balance. Exiting.")
            return
    except Exception as e:
        logger.error(f"Failed to connect or get initial balance: {e}")
        return

    # Create strategy instance using the factory function
    strategy = get_strategy(strategy_name, client, interval)
    if not strategy:
        logger.error(f"Could not create strategy instance for '{strategy_name}'. Exiting.")
        return

    # --- Main Loop ---
    start_time = datetime.now()
    final_balance = initial_balance # Initialize final balance

    try:
        while True:
            logger.info(f"--- Running strategy cycle ({datetime.now()}) ---")
            try:
                strategy.execute()
            except Exception as e:
                logger.error(f"Error during strategy execution: {e}", exc_info=True)
                # Decide if error is critical enough to stop the bot
                time.sleep(60) # Wait a minute before retrying after an error

            # Wait for the next candle/interval
            wait_seconds = utils.calculate_wait_time(interval)
            logger.info(f"Waiting {wait_seconds:.2f} seconds for next {interval} candle")
            time.sleep(wait_seconds)

    except KeyboardInterrupt:
        logger.info("Bot stopped manually (KeyboardInterrupt).")
    
    finally:
        # --- Log Results ---
        logger.info("--- Bot shutting down. Fetching final balance... ---")
        try:
            final_balance_info = client.get_account_balance()
            if final_balance_info and 'USDT' in final_balance_info:
                 final_balance = final_balance_info['USDT']
                 logger.info(f"Final Account USDT balance: {final_balance:.2f}")
            else:
                 logger.warning("Could not retrieve final USDT balance. Using last known or initial value for PnL.")
                 # Use the value from the start if final couldn't be fetched
                 final_balance = initial_balance 

            pnl = final_balance - initial_balance
            logger.info(f"Trading Session P&L: {pnl:.2f} USDT")
            
            # Log results to CSV
            end_time = datetime.now()
            log_results_to_csv(start_time, end_time, strategy_name, client.symbol, interval, initial_balance, final_balance)
            
        except Exception as e:
            logger.error(f"Error fetching final balance or logging results: {e}")
            
        logger.info("--- Bot Shutdown Complete ---")

def main():
    """Main entry point"""
    # Create log directory
    os.makedirs('logs', exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    parser.add_argument('--strategy', type=str, default='proyecto', choices=['sma', 'rsi', 'proyecto', 'ml'],
                        help='Trading strategy (sma, rsi, proyecto, ml)')
    parser.add_argument('--interval', type=str, default='15m',
                        help='Trading interval (e.g., 1m, 5m, 15m, 1h)')
    parser.add_argument('--once', action='store_true',
                        help='Run the strategy once instead of continuously')
    parser.add_argument('--report', action='store_true',
                        help='Generate a PnL report from trade history')
    
    args = parser.parse_args()
    
    if args.report:
        # Generate PnL report
        trades = utils.get_trade_history()
        total_pnl = utils.calculate_pnl(trades)
        utils.plot_trade_history(trades)
        logger.info(f"Total PnL: {total_pnl}")
    else:
        # Run the trading bot
        run_bot(args.strategy, args.interval)

if __name__ == "__main__":
    main() 