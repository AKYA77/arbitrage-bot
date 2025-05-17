import requests
import websockets
import json
import hmac
import hashlib
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from core.config import load_api_keys

logger = logging.getLogger("mexc_wrapper")

class MEXCWrapper:
    def __init__(self):
        """
        Initialize the MEXC exchange wrapper with API credentials if available.
        Falls back to public API if no credentials are provided.
        """
        self.base_url = 'https://api.mexc.com/api/v3'
        self.ws_url = 'wss://wbs.mexc.com/ws'
        
        api_keys = load_api_keys().get("mexc", {})
        self.api_key = api_keys.get("api_key", "")
        self.api_secret = api_keys.get("api_secret", "")
        
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
            
            # Format symbol for MEXC (remove '/')
            formatted_symbol = symbol.replace('/', '')
            
            # Make API request
            response = requests.get(
                f"{self.base_url}/depth",
                params={'symbol': formatted_symbol, 'limit': depth},
                timeout=10  # Add timeout for reliability
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Validate response format
            if 'bids' not in data or 'asks' not in data:
                raise Exception(f"Invalid response format from MEXC API: {data}")
            
            # Format order book
            order_book = {
                'bids': data['bids'],
                'asks': data['asks']
            }
            
            # Update cache
            self.market_cache[cache_key] = {
                'timestamp': time.time(),
                'data': order_book
            }
            
            return order_book
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error fetching order book for {symbol} on MEXC: {str(e)}")
        except ValueError as e:
            raise Exception(f"JSON parsing error for {symbol} on MEXC: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching order book for {symbol} on MEXC: {str(e)}")
            
    def get_fee_rate(self) -> float:
        """
        Get the trading fee rate for this exchange.
        
        Returns:
            float: Fee rate as a decimal (e.g., 0.002 for 0.2%)
        """
        try:
            # Try to get fee from API if authenticated
            if self.api_key and self.api_secret:
                # Implementation would depend on MEXC API
                pass
                
            # Default fee if not authenticated or API call fails
            return 0.002  # 0.2% is MEXC's standard fee
        except:
            return 0.002  # Default to 0.2%
            
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
        if not self.api_key or not self.api_secret:
            raise Exception("API keys required for trading")
            
        try:
            # Format symbol for MEXC (remove '/')
            formatted_symbol = symbol.replace('/', '')
            
            # Prepare parameters
            params = {
                'symbol': formatted_symbol,
                'side': side.upper(),
                'type': 'MARKET',
                'quantity': str(amount),
                'timestamp': str(int(time.time() * 1000))
            }
            
            # Generate signature
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            # Make API request
            headers = {'X-MEXC-APIKEY': self.api_key}
            response = requests.post(
                f"{self.base_url}/order",
                headers=headers,
                data=params,
                timeout=10
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            return {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'filled': amount,  # Assuming full fill for market orders
                'cost': amount * float(data.get('price', 0)),
                'status': 'closed',
                'info': data
            }
        except Exception as e:
            raise Exception(f"Error creating {side} order for {symbol}: {str(e)}")
    
    def get_websocket_url(self) -> str:
        """
        Get the WebSocket URL for this exchange.
        
        Returns:
            str: WebSocket URL
        """
        return self.ws_url
    
    def get_websocket_subscribe_message(self) -> str:
        """
        Get the WebSocket subscription message for order book updates.
        
        Returns:
            str: JSON subscription message
        """
        # Common trading pairs
        pairs = ["BTC_USDT", "ETH_USDT", "ETH_BTC", "LTC_USDT", "LTC_BTC", "XRP_USDT", "XRP_BTC"]
        
        # Format subscription message
        subscriptions = []
        for pair in pairs:
            subscriptions.append({
                "symbol": pair,
                "topic": "depth5"  # 5 levels of depth
            })
        
        message = {
            "method": "SUBSCRIPTION",
            "params": subscriptions
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
            if 'channel' in data and data['channel'] == 'push.depth5':
                if 'data' in data and isinstance(data['data'], dict):
                    result = data['data']
                    if 'symbol' in result:
                        # Format symbol with '/'
                        symbol = result['symbol'].replace('_', '/')
                        
                        # Extract bids and asks
                        bids = result.get('bids', [])
                        asks = result.get('asks', [])
                        
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
            logger.error(f"Error parsing MEXC WebSocket message: {str(e)}")
            return []
    
    async def create_websocket_connection(self):
        """
        Create a WebSocket connection to the exchange.
        
        Returns:
            websockets.WebSocketClientProtocol: WebSocket connection
        """
        return await websockets.connect(self.get_websocket_url())
