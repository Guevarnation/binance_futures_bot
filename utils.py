import os
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import time
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_trade_log_file():
    """Create a trade log file if it doesn't exist"""
    os.makedirs('logs', exist_ok=True)
    trade_log_path = 'logs/trade_log.json'
    if not os.path.exists(trade_log_path):
        with open(trade_log_path, 'w') as f:
            json.dump([], f)
        logger.info(f"Created new trade log at {trade_log_path}")
    return trade_log_path

def log_trade(trade_data):
    """Log a trade to JSON file"""
    trade_log_path = create_trade_log_file()
    try:
        with open(trade_log_path, 'r') as f:
            trades = json.load(f)
        
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
            
        trades.append(trade_data)
        
        with open(trade_log_path, 'w') as f:
            json.dump(trades, f, indent=2)
            
        logger.info(f"Logged trade: {trade_data}")
        return True
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")
        return False

def get_trade_history():
    """Get trade history from log file"""
    trade_log_path = create_trade_log_file()
    try:
        with open(trade_log_path, 'r') as f:
            trades = json.load(f)
        return trades
    except Exception as e:
        logger.error(f"Failed to get trade history: {e}")
        return []

def calculate_pnl(trades):
    """Calculate profit and loss from trade history"""
    if not trades:
        logger.warning("No trades to calculate PnL")
        return 0
    
    total_pnl = 0
    for trade in trades:
        if 'pnl' in trade:
            total_pnl += float(trade['pnl'])
    
    logger.info(f"Total PnL: {total_pnl:.4f}")
    return total_pnl

def plot_trade_history(trades):
    """Plot trade history"""
    if not trades:
        logger.warning("No trades to plot")
        return False
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate cumulative PnL
        if 'pnl' in df.columns:
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['cumulative_pnl'])
            plt.title('Cumulative PnL Over Time')
            plt.xlabel('Time')
            plt.ylabel('PnL')
            plt.grid(True)
            
            # Save plot
            os.makedirs('plots', exist_ok=True)
            plot_path = f'plots/pnl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(plot_path)
            logger.info(f"PnL plot saved to {plot_path}")
            return True
        else:
            logger.warning("No PnL data in trades")
            return False
    except Exception as e:
        logger.error(f"Failed to plot trade history: {e}")
        return False

def wait_for_next_candle(interval):
    """Wait until the start of the next candle for a given interval"""
    interval_seconds = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '6h': 21600,
        '12h': 43200,
        '1d': 86400
    }
    
    seconds = interval_seconds.get(interval)
    if not seconds:
        logger.warning(f"Unknown interval: {interval}, defaulting to 5m")
        seconds = 300
    
    current_time = time.time()
    next_candle = current_time - (current_time % seconds) + seconds
    wait_time = next_candle - current_time
    
    logger.info(f"Waiting {wait_time:.2f} seconds for next {interval} candle")
    time.sleep(wait_time)
    return True

def calculate_lot_size(account_balance, risk_per_trade, stop_loss_percentage, leverage):
    """
    Calculate appropriate lot size based on account balance and risk
    
    Args:
        account_balance: Total account balance in USDT
        risk_per_trade: Percentage of account balance to risk per trade (e.g., 1 for 1%)
        stop_loss_percentage: Stop loss percentage (e.g., 1 for 1%)
        leverage: Trading leverage
        
    Returns:
        Lot size in the base currency
    """
    if not account_balance or stop_loss_percentage == 0:
        return 0
    
    # Calculate max amount to risk per trade
    risk_amount = account_balance * (risk_per_trade / 100)
    
    # Calculate position size in USDT
    position_size = (risk_amount / (stop_loss_percentage / 100)) * leverage
    
    logger.info(f"Calculated position size: {position_size:.4f} USDT (Leverage: {leverage}x)")
    return position_size

def calculate_wait_time(interval):
    """Calculate the number of seconds to wait until the next candle.

    Args:
        interval (str): The candle interval (e.g., '1m', '5m', '15m', '1h', '4h').

    Returns:
        int: The number of seconds to wait.
    """
    now = datetime.now(timezone.utc)
    
    # Convert interval string to timedelta
    if 'm' in interval:
        minutes = int(interval.replace('m', ''))
        delta = timedelta(minutes=minutes)
        # Calculate the timestamp of the start of the current candle
        current_candle_start_ts = math.floor(now.timestamp() / (minutes * 60)) * (minutes * 60)
    elif 'h' in interval:
        hours = int(interval.replace('h', ''))
        delta = timedelta(hours=hours)
        # Calculate the timestamp of the start of the current candle
        current_candle_start_ts = math.floor(now.timestamp() / (hours * 3600)) * (hours * 3600)
    # Add more interval types (e.g., 'd' for day) if needed
    else:
        logger.error(f"Unsupported interval format: {interval}. Defaulting to 60 seconds wait.")
        return 60
        
    # Calculate the timestamp of the start of the next candle
    next_candle_start_ts = current_candle_start_ts + delta.total_seconds()
    
    # Calculate seconds to wait
    wait_seconds = next_candle_start_ts - now.timestamp()
    
    # Ensure wait time is not negative (can happen due to slight timing issues)
    # Also add a small buffer (e.g., 1 second) to ensure the next candle has started
    wait_seconds = max(0, wait_seconds) + 1 

    return wait_seconds 