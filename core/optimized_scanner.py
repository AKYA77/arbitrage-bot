import os
import json
import time
import logging
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("optimized_scanner")

class OptimizedArbitrageScanner:
    """
    Optimized arbitrage scanner with fee and slippage awareness.
    """
    
    def __init__(self, exchange_wrappers: Dict):
        """
        Initialize the optimized arbitrage scanner.
        
        Args:
            exchange_wrappers: Dictionary of exchange name to wrapper instance
        """
        self.exchange_wrappers = exchange_wrappers
        
        # Default configuration
        self.config = {
            'min_profit_threshold': 0.5,  # Minimum profit percentage after fees
            'safety_margin': 0.2,         # Additional safety margin for execution
            'max_slippage': 0.3,          # Maximum allowed slippage percentage
            'order_book_depth': 15,       # Depth to analyze in order book
            'max_position_size': 1000,    # Maximum position size in base currency
            'min_position_size': 10,      # Minimum position size in base currency
            'parallel_scanning': True,    # Whether to scan exchanges in parallel
        }
        
        # Cache for exchange fees
        self.fee_cache = {}
        
        # Cache for discovered loops
        self.loop_cache = self._load_loop_cache()
        
        logger.info("Optimized Arbitrage Scanner initialized")
    
    def _load_loop_cache(self) -> Dict:
        """Load cached loops from disk."""
        try:
            cache_path = os.path.join("data", "cached_loops.json")
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                
                # Ensure the cache has the expected structure
                if not isinstance(cache, dict):
                    cache = {"loops": [], "version": 1, "last_updated": time.time()}
                elif "loops" not in cache:
                    cache["loops"] = []
                
                return cache
            
            return {"loops": [], "version": 1, "last_updated": time.time()}
        except Exception as e:
            logger.error(f"Error loading loop cache: {str(e)}")
            return {"loops": [], "version": 1, "last_updated": time.time()}
    
    def get_exchange_fee(self, exchange_name: str) -> float:
        """Get exchange fee rate with caching."""
        if exchange_name not in self.fee_cache:
            try:
                self.fee_cache[exchange_name] = self.exchange_wrappers[exchange_name].get_fee_rate()
            except:
                # Default fee if not available
                self.fee_cache[exchange_name] = 0.001  # 0.1%
        return self.fee_cache[exchange_name]
    
    def scan_for_opportunities(self, loops: Optional[List[List[str]]] = None) -> List[Dict]:
        """
        Scan for arbitrage opportunities across all exchanges and loops.
        
        Args:
            loops: Optional list of trading loops to check, if None uses default loops
            
        Returns:
            List of profitable opportunities
        """
        # If no loops provided, use some default triangular loops
        if loops is None:
            loops = [
                ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
                ["ETH/USDT", "XRP/ETH", "XRP/USDT"],
                ["BTC/USDT", "LTC/BTC", "LTC/USDT"]
            ]
        
        opportunities = []
        
        if self.config['parallel_scanning']:
            # Parallel scanning for better performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(self.exchange_wrappers) * len(loops))) as executor:
                futures = []
                for exchange_name, wrapper in self.exchange_wrappers.items():
                    for loop in loops:
                        futures.append(
                            executor.submit(
                                self.check_loop_profitability, 
                                exchange_name, 
                                wrapper, 
                                loop
                            )
                        )
                
                for future in futures:
                    result = future.result()
                    if result:
                        opportunities.append(result)
        else:
            # Sequential scanning
            for exchange_name, wrapper in self.exchange_wrappers.items():
                for loop in loops:
                    result = self.check_loop_profitability(exchange_name, wrapper, loop)
                    if result:
                        opportunities.append(result)
        
        # Sort by profit
        return sorted(
            opportunities, 
            key=lambda x: x.get('adjusted_profit_percent', 0), 
            reverse=True
        )
    
    def check_loop_profitability(self, exchange_name: str, wrapper: Any, loop: List[str]) -> Optional[Dict]:
        """
        Check profitability of a single loop on a specific exchange.
        
        Args:
            exchange_name: Name of the exchange
            wrapper: Exchange wrapper instance
            loop: Trading loop to check
            
        Returns:
            Opportunity details if profitable, None otherwise
        """
        try:
            # Get order books with sufficient depth
            depth = self.config['order_book_depth']
            ob1 = wrapper.get_order_book(loop[0], depth=depth)
            ob2 = wrapper.get_order_book(loop[1], depth=depth)
            ob3 = wrapper.get_order_book(loop[2], depth=depth)
            
            # Validate order books
            if not self._validate_order_books([ob1, ob2, ob3], loop):
                return None
            
            # Get exchange fee
            fee_rate = self.get_exchange_fee(exchange_name)
            
            # Calculate optimal position size based on liquidity
            optimal_size = self._calculate_optimal_size(ob1, ob2, ob3)
            
            # Apply position size limits
            position_size = min(
                max(optimal_size, self.config['min_position_size']),
                self.config['max_position_size']
            )
            
            # Simulate execution with slippage and fees
            first_trade_result = self._simulate_market_buy(ob1, position_size, fee_rate)
            if not first_trade_result['success']:
                return None
                
            second_trade_result = self._simulate_market_sell(
                ob2, 
                first_trade_result['resulting_amount'], 
                fee_rate
            )
            if not second_trade_result['success']:
                return None
                
            final_trade_result = self._simulate_market_sell(
                ob3, 
                second_trade_result['resulting_amount'], 
                fee_rate
            )
            if not final_trade_result['success']:
                return None
            
            # Calculate profit with all fees included
            final_amount = final_trade_result['resulting_amount']
            profit = final_amount - position_size
            profit_pct = (profit / position_size) * 100
            
            # Apply safety margin for execution latency
            adjusted_profit_pct = profit_pct - self.config['safety_margin']
            
            # Check if profitable after all adjustments
            if adjusted_profit_pct > self.config['min_profit_threshold']:
                return {
                    'exchange': exchange_name,
                    'loop': loop,
                    'position_size': position_size,
                    'raw_profit_percent': profit_pct,
                    'adjusted_profit_percent': adjusted_profit_pct,
                    'timestamp': time.time(),
                    'trade_simulation': {
                        'first_trade': first_trade_result,
                        'second_trade': second_trade_result,
                        'final_trade': final_trade_result
                    }
                }
        
        except Exception as e:
            logger.error(f"Error checking loop {loop} on {exchange_name}: {str(e)}")
            return None
            
        return None
    
    def _validate_order_books(self, order_books: List[Dict], loop: List[str]) -> bool:
        """Validate order books have sufficient data."""
        for i, ob in enumerate(order_books):
            if not ob or 'bids' not in ob or 'asks' not in ob:
                return False
            if not ob['bids'] or not ob['asks']:
                return False
        return True
    
    def _calculate_optimal_size(self, ob1: Dict, ob2: Dict, ob3: Dict) -> float:
        """
        Calculate optimal position size based on order book liquidity.
        
        Returns the maximum amount that can be executed with acceptable slippage.
        """
        # Calculate available liquidity at each step
        max_buy_1 = sum(float(qty) * float(price) for price, qty in ob1['asks'][:5])
        max_sell_2 = sum(float(qty) for price, qty in ob2['bids'][:5])
        max_sell_3 = sum(float(qty) for price, qty in ob3['bids'][:5])
        
        # Convert to equivalent base currency amounts
        max_buy_1_base = max_buy_1
        max_sell_2_base = max_sell_2 * float(ob2['bids'][0][0])
        max_sell_3_base = max_sell_3 * float(ob3['bids'][0][0])
        
        # Find the limiting factor (minimum liquidity across all steps)
        return min(max_buy_1_base, max_sell_2_base, max_sell_3_base) * 0.7  # Use 70% of available liquidity
    
    def _simulate_market_buy(self, order_book: Dict, amount: float, fee_rate: float) -> Dict:
        """
        Simulate a market buy with slippage and fees.
        
        Args:
            order_book: Order book data
            amount: Amount to spend
            fee_rate: Exchange fee rate
            
        Returns:
            Dict with simulation results
        """
        remaining = amount
        acquired = 0
        avg_price = 0
        total_spent = 0
        
        for price, quantity in order_book['asks']:
            price_float = float(price)
            quantity_float = float(quantity)
            
            max_spend_at_level = price_float * quantity_float
            spend_at_level = min(remaining, max_spend_at_level)
            
            qty_acquired = spend_at_level / price_float
            
            total_spent += spend_at_level
            acquired += qty_acquired
            remaining -= spend_at_level
            
            if remaining <= 0:
                break
        
        if remaining > 0 and remaining / amount > self.config['max_slippage'] / 100:
            # Too much slippage
            return {'success': False, 'reason': 'excessive_slippage'}
        
        # Apply trading fee
        acquired_after_fee = acquired * (1 - fee_rate)
        
        return {
            'success': True,
            'initial_amount': amount,
            'resulting_amount': acquired_after_fee,
            'average_price': total_spent / acquired if acquired > 0 else 0,
            'slippage_percent': ((total_spent / acquired if acquired > 0 else 0) / float(order_book['asks'][0][0]) - 1) * 100 if acquired > 0 else 0,
            'fee_amount': acquired * fee_rate
        }
    
    def _simulate_market_sell(self, order_book: Dict, amount: float, fee_rate: float) -> Dict:
        """
        Simulate a market sell with slippage and fees.
        
        Args:
            order_book: Order book data
            amount: Amount to sell
            fee_rate: Exchange fee rate
            
        Returns:
            Dict with simulation results
        """
        remaining = amount
        received = 0
        avg_price = 0
        
        for price, quantity in order_book['bids']:
            price_float = float(price)
            quantity_float = float(quantity)
            
            sell_at_level = min(remaining, quantity_float)
            
            received_at_level = sell_at_level * price_float
            
            received += received_at_level
            remaining -= sell_at_level
            
            if remaining <= 0:
                break
        
        if remaining > 0 and remaining / amount > self.config['max_slippage'] / 100:
            # Too much slippage
            return {'success': False, 'reason': 'excessive_slippage'}
        
        # Apply trading fee
        received_after_fee = received * (1 - fee_rate)
        
        return {
            'success': True,
            'initial_amount': amount,
            'resulting_amount': received_after_fee,
            'average_price': received / (amount - remaining) if amount > remaining else 0,
            'slippage_percent': (1 - (received / (amount - remaining) if amount > remaining else 0) / float(order_book['bids'][0][0])) * 100 if amount > remaining else 0,
            'fee_amount': received * fee_rate
        }
