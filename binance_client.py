import os
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from dotenv import load_dotenv
# import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BinanceFuturesClient:
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_KEY')
        testnet = os.getenv('TESTNET', 'true').lower() == 'true'
        
        if not api_key or not api_secret:
            raise ValueError("API key and secret must be provided in .env file")
        
        self.client = Client(api_key, api_secret, testnet=testnet)
        logger.info(f"Initialized Binance Futures client (Testnet: {testnet})")
        
        # Default trading parameters
        self.symbol = os.getenv('SYMBOL', 'BTCUSDT')
        self.leverage = int(os.getenv('LEVERAGE', '5'))
        self.quantity = float(os.getenv('QUANTITY', '0.001'))
        
        # Set leverage
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            logger.info(f"Set leverage to {self.leverage}x for {self.symbol}")
        except BinanceAPIException as e:
            logger.error(f"Failed to set leverage: {e}")
    
    def get_account_balance(self):
        """Get futures account balance"""
        try:
            account = self.client.futures_account()
            logger.info(f"Raw futures_account() response: {account}")
            
            balances = {}
            if 'assets' in account:
                for asset in account['assets']:
                    balance_key = None
                    if 'balance' in asset:
                        balance_key = 'balance'
                    elif 'walletBalance' in asset:
                        balance_key = 'walletBalance'
                    elif 'availableBalance' in asset:
                        balance_key = 'availableBalance'
                    elif 'free' in asset:
                        balance_key = 'free'
                    
                    if balance_key and 'asset' in asset:
                        try:
                            balances[asset['asset']] = float(asset[balance_key])
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert balance for asset {asset.get('asset')} to float: {asset.get(balance_key)}")
                    else:
                        logger.warning(f"Could not find balance key or asset name in asset data: {asset}")
            else:
                logger.warning("'assets' key not found in futures_account response.")

            logger.info(f"Processed Account balances: {balances}")
            return balances
        except BinanceAPIException as e:
            logger.error(f"Failed to get account balance: {e}")
            return {}
    
    def get_market_price(self, symbol=None):
        """Get current market price for a symbol"""
        symbol = symbol or self.symbol
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logger.info(f"Current {symbol} price: {price}")
            return price
        except BinanceAPIException as e:
            logger.error(f"Failed to get market price: {e}")
            return None
    
    def place_order(self, side, quantity=None, symbol=None, order_type='MARKET', price=None, 
                    time_in_force='GTC', reduce_only=False, close_position=False):
        """Place a futures order"""
        symbol = symbol or self.symbol
        quantity = quantity or self.quantity
        
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'reduceOnly': reduce_only,
            'closePosition': close_position,
        }
        
        if order_type == 'MARKET':
            order_params['quantity'] = quantity
        elif order_type == 'LIMIT':
            if not price:
                raise ValueError("Price must be provided for LIMIT orders")
            order_params['quantity'] = quantity
            order_params['price'] = price
            order_params['timeInForce'] = time_in_force
        
        try:
            order = self.client.futures_create_order(**order_params)
            logger.info(f"Order placed: {order}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def place_market_buy(self, quantity=None, symbol=None):
        """Place a market buy order"""
        return self.place_order('BUY', quantity, symbol, 'MARKET')
    
    def place_market_sell(self, quantity=None, symbol=None):
        """Place a market sell order"""
        return self.place_order('SELL', quantity, symbol, 'MARKET')
    
    def place_limit_buy(self, price, quantity=None, symbol=None):
        """Place a limit buy order"""
        return self.place_order('BUY', quantity, symbol, 'LIMIT', price)
    
    def place_limit_sell(self, price, quantity=None, symbol=None):
        """Place a limit sell order"""
        return self.place_order('SELL', quantity, symbol, 'LIMIT', price)
    
    def place_take_profit_order(self, entry_price, side, quantity=None, symbol=None, profit_percentage=None):
        """Place a take profit order"""
        symbol = symbol or self.symbol
        quantity = quantity or self.quantity
        profit_percentage = profit_percentage or float(os.getenv('PROFIT_PERCENTAGE', '1.5'))
        
        if side == 'BUY':
            take_profit_price = entry_price * (1 + profit_percentage / 100)
            opposite_side = 'SELL'
        else:
            take_profit_price = entry_price * (1 - profit_percentage / 100)
            opposite_side = 'BUY'
        
        logger.info(f"Setting take profit at {take_profit_price} ({profit_percentage}% from {entry_price})")
        return self.place_order(opposite_side, quantity, symbol, 'LIMIT', take_profit_price, reduce_only=True)
    
    def place_stop_loss_order(self, entry_price, side, quantity=None, symbol=None, stop_loss_percentage=None):
        """Place a stop loss order"""
        symbol = symbol or self.symbol
        quantity = quantity or self.quantity
        stop_loss_percentage = stop_loss_percentage or float(os.getenv('STOP_LOSS_PERCENTAGE', '1.0'))
        
        if side == 'BUY':
            stop_price = entry_price * (1 - stop_loss_percentage / 100)
            opposite_side = 'SELL'
        else:
            stop_price = entry_price * (1 + stop_loss_percentage / 100)
            opposite_side = 'BUY'
            
        logger.info(f"Setting stop loss at {stop_price} ({stop_loss_percentage}% from {entry_price})")
        
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=opposite_side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                reduceOnly=True
            )
            logger.info(f"Stop loss order placed: {order}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Failed to place stop loss order: {e}")
            return None
    
    def get_open_positions(self):
        """Get all open positions"""
        try:
            positions = self.client.futures_position_information()
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            logger.info(f"Open positions: {open_positions}")
            return open_positions
        except BinanceAPIException as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
    
    def get_open_orders(self, symbol=None):
        """Get all open orders for a symbol"""
        symbol = symbol or self.symbol
        try:
            orders = self.client.futures_get_open_orders(symbol=symbol)
            logger.info(f"Open orders for {symbol}: {orders}")
            return orders
        except BinanceAPIException as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def cancel_all_open_orders(self, symbol=None):
        """Cancel all open orders for a symbol"""
        symbol = symbol or self.symbol
        try:
            result = self.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"Cancelled all open orders for {symbol}: {result}")
            return result
        except BinanceAPIException as e:
            logger.error(f"Failed to cancel open orders: {e}")
            return None
    
    def close_position(self, symbol=None):
        """Close a position for a symbol"""
        symbol = symbol or self.symbol
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for position in positions:
                if float(position['positionAmt']) != 0:
                    side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
                    quantity = abs(float(position['positionAmt']))
                    self.place_order(side, quantity, symbol, 'MARKET', close_position=True)
                    logger.info(f"Closed position for {symbol}")
                    return True
            logger.info(f"No position to close for {symbol}")
            return False
        except BinanceAPIException as e:
            logger.error(f"Failed to close position: {e}")
            return False 