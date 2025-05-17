import ccxt
import websockets
import json
import hmac
import hashlib
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from core.config import load_api_keys

logger = logging.getLogger("gateio_wrapper")

class GateIOWrapper:
    def __init__(self):
        """
        Initialize the GateIO exchange wrapper with API credentials if available.
        Falls back to public API if no credentials are provided.
        """
        api_keys = load_api_keys().get("gateio", {})
        self.api_key = api_keys.get("api_key", "")
        self.api_secret = api_keys.get("api_secret", "")
        
        self.client = ccxt.gateio({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,  # Enable built-in rate limiting
        })
        
        # WebSocket URLs
        self.ws_public_url = "wss://api.gateio.ws/ws/v4/"
        self.ws_private_url = "wss://api.gateio.ws/ws/v4/"
        
        # Cache for market data
        self.market_cache = {}
        
    def get_order_book(self, symbol: str, depth: int = 5) -> Dict:
        """
        Get the order book for a specific trading pair.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            depth (int): Depth of the order book to retrieve
            
        Returns:
            dict: Order book with 'bids' and 'asks' keys
            
        Raises:
            Exception: If there's an error fetching the order book
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{depth}"
            if cache_key in self.market_cache:
                cache_entry = self.market_cache[cache_key]
                # Use cache if it's less than 2 seconds old
                if time.time() - cache_entry['timestamp'] < 2:
                    return cache_entry['data']
            
            # Fetch from API if not in cache or cache is stale
            order_book = self.client.fetch_order_book(symbol, depth)
            
            # Update cache
            self.market_cache[cache_key] = {
                'timestamp': time.time(),
                'data': order_book
            }
            
            return order_book
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Error fetching order book for {symbol} on GateIO: {str(e)}")
            
    def get_fee_rate(self) -> float:
        """
        Get the trading fee rate for this exchange.
        
        Returns:
            float: Fee rate as a decimal (e.g., 0.001 for 0.1%)
        """
        try:
            # Try to get actual fee from exchange
            markets = self.client.fetch_markets()
            if markets and len(markets) > 0:
                # Use the first market's taker fee as an approximation
                return markets[0].get('taker', 0.001)
        except:
            pass
        
        # Default fee if unable to get from exchange
        return 0.001  # 0.1%
        
    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """
        Create a market order on the exchange.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): 'buy' or 'sell'
            amount (float): Amount to buy or sell
            
        Returns:
            dict: Order details
        """
        try:
            return self.client.create_market_order(symbol, side, amount)
        except Exception as e:
            raise Exception(f"Error creating {side} order for {symbol}: {str(e)}")
    
    def get_websocket_url(self) -> str:
        """
        Get the WebSocket URL for this exchange.
        
        Returns:
            str: WebSocket URL
        """
        return self.ws_public_url
    
    def get_websocket_subscribe_message(self) -> str:
        """
        Get the WebSocket subscription message for order book updates.
        
        Returns:
            str: JSON subscription message
        """
        # Subscribe to spot order book updates for common trading pairs
        pairs = ["BTC_USDT", "ETH_USDT", "ETH_BTC", "LTC_USDT", "LTC_BTC", "XRP_USDT", "XRP_BTC"]
        
        channels = []
        for pair in pairs:
            channels.append({
                "channel": "spot.order_book",
                "payload": [pair, "5", "100ms"]  # pair, depth, interval
            })
        
        message = {
            "time": int(time.time()),
            "id": 12345,
            "channel": "spot.order_book",
            "event": "subscribe",
            "payload": channels
        }
        
        return json.dumps(message)
    
    def parse_websocket_message(self, message: str) -> List[Dict]:
        """
        Parse a WebSocket message from the exchange.
        
        Args:
            message (str): WebSocket message
            
        Returns:
            List[Dict]: List of parsed updates
        """
        try:
            data = json.loads(message)
            
            # Check if it's an order book update
            if 'channel' in data and data['channel'] == 'spot.order_book':
                if 'event' in data and data['event'] == 'update':
                    # Extract the trading pair
                    if 'result' in data and isinstance(data['result'], dict):
                        result = data['result']
                        if 's' in result:  # Symbol
                            symbol = result['s'].replace('_', '/')
                            
                            # Extract bids and asks
                            bids = result.get('b', [])  # Bids
                            asks = result.get('a', [])  # Asks
                            
                            # Format as order book
                            return [{
                                'type': 'orderbook',
                                'symbol': symbol,
                                'bids': bids,
                                'asks': asks,
                                'timestamp': time.time()
                            }]
            
            # Return empty list if not an order book update
            return []
        except Exception as e:
            logger.error(f"Error parsing GateIO WebSocket message: {str(e)}")
            return []
    
    async def create_websocket_connection(self):
        """
        Create a WebSocket connection to the exchange.
        
        Returns:
            websockets.WebSocketClientProtocol: WebSocket connection
        """
        return await websockets.connect(self.get_websocket_url())
