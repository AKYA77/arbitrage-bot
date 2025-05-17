import asyncio
import websockets
import json
import time
import logging
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from dataclasses import dataclass, field

from core.graph_algorithms import ArbitrageGraph, JohnsonAlgorithm, BellmanFordAlgorithm, YenKShortestPaths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced_scanner")

class AdvancedArbitrageScanner:
    """
    Advanced arbitrage scanner with cutting-edge algorithms and WebSocket integration.
    """
    
    def __init__(self, exchange_wrappers: Dict, config: Dict = None):
        """
        Initialize the advanced arbitrage scanner.
        
        Args:
            exchange_wrappers: Dictionary of exchange name to wrapper instance
            config: Configuration parameters
        """
        self.exchange_wrappers = exchange_wrappers
        self.config = config or {
            'min_profit_threshold': 0.5,  # Minimum profit percentage after fees
            'safety_margin': 0.2,         # Additional safety margin for execution
            'max_slippage': 0.3,          # Maximum allowed slippage percentage
            'order_book_depth': 15,       # Depth to analyze in order book
            'max_position_size': 1000,    # Maximum position size in base currency
            'min_position_size': 10,      # Minimum position size in base currency
            'max_execution_time': 2000,   # Maximum allowed execution time in ms
            'parallel_scanning': True,    # Whether to scan exchanges in parallel
            'risk_reward_ratio': 3.0,     # Minimum risk-reward ratio
            'max_concurrent_trades': 3,   # Maximum number of concurrent trades
            'algorithm': 'johnson',       # Algorithm to use: 'johnson', 'bellman-ford', 'yen-k-paths'
            'k_paths': 5,                 # Number of paths to find with Yen's algorithm
            'incremental': True,          # Whether to use incremental updates
            'subgraph_radius': 2,         # Radius for subgraph extraction in incremental updates
            'depth_threshold': 0.1,       # Minimum depth threshold for pruning
            'volatility_threshold': 0.5,  # Maximum volatility for considering an opportunity
        }
        
        # Initialize the exchange rate graph
        self.graph = ArbitrageGraph()
        
        # Initialize trade history and performance metrics
        self.trade_history = []
        self.performance_metrics = {
            'total_profit': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        # Cache for exchange fees
        self.fee_cache = {}
        
        # Initialize WebSocket connection managers
        self.ws_managers = {}
        
        # Initialize the event loop for async operations
        self.loop = asyncio.new_event_loop()
        
        # Flag to track if WebSockets are running
        self.ws_running = False
        
        # Thread for running the event loop
        self.event_loop_thread = None
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Circuit breaker state
        self.circuit_breakers = {ex: {'failures': 0, 'open': False, 'last_failure': 0} 
                               for ex in exchange_wrappers.keys()}
        
        # Cache for discovered loops
        self.loop_cache = self._load_loop_cache()
        
        logger.info("Advanced Arbitrage Scanner initialized")
    
    def _load_loop_cache(self) -> Dict:
        """Load cached loops from disk with versioning and timestamps."""
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
                elif "version" not in cache:
                    cache["version"] = 1
                elif "last_updated" not in cache:
                    cache["last_updated"] = time.time()
                
                return cache
            
            return {"loops": [], "version": 1, "last_updated": time.time()}
        except Exception as e:
            logger.error(f"Error loading loop cache: {str(e)}")
            return {"loops": [], "version": 1, "last_updated": time.time()}
    
    async def _async_save_cache(self):
        """Save cache asynchronously to avoid blocking."""
        try:
            cache_path = os.path.join("data", "cached_loops.json")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Update timestamp
            self.loop_cache["last_updated"] = time.time()
            
            # Increment version
            self.loop_cache["version"] = self.loop_cache.get("version", 0) + 1
            
            # Write to temporary file first
            temp_path = cache_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(self.loop_cache, f, indent=2)
            
            # Rename to actual file (atomic operation)
            os.replace(temp_path, cache_path)
            
            logger.debug("Loop cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving loop cache: {str(e)}")
    
    def add_loop_to_cache(self, loop: List[str], metadata: Dict = None):
        """
        Add a new loop to the cache if it doesn't exist.
        
        Args:
            loop: List of trading pairs forming a loop
            metadata: Additional metadata about the loop
        """
        # Generate a unique ID for the loop
        loop_id = "-".join(loop)
        
        # Check if loop already exists
        existing_ids = {l.get("id") for l in self.loop_cache["loops"]}
        if loop_id in existing_ids:
            return
        
        # Create loop entry with timestamp
        loop_entry = {
            "id": loop_id,
            "pairs": loop,
            "discovered_at": time.time(),
            "metadata": metadata or {}
        }
        
        # Add to cache
        self.loop_cache["loops"].append(loop_entry)
        
        # Save cache asynchronously
        asyncio.create_task(self._async_save_cache())
    
    def get_exchange_fee(self, exchange_name: str) -> float:
        """Get exchange fee rate with caching."""
        if exchange_name not in self.fee_cache:
            try:
                self.fee_cache[exchange_name] = self.exchange_wrappers[exchange_name].get_fee_rate()
            except:
                # Default fee if not available
                self.fee_cache[exchange_name] = 0.001  # 0.1%
        return self.fee_cache[exchange_name]
    
    async def start_websocket_feeds(self):
        """Start WebSocket feeds for all exchanges."""
        if self.ws_running:
            return
        
        self.ws_running = True
        
        for exchange_name, wrapper in self.exchange_wrappers.items():
            if hasattr(wrapper, 'get_websocket_url'):
                try:
                    url = wrapper.get_websocket_url()
                    self.ws_managers[exchange_name] = {
                        'url': url,
                        'task': asyncio.create_task(self._websocket_manager(exchange_name, url))
                    }
                    logger.info(f"Started WebSocket for {exchange_name}")
                except Exception as e:
                    logger.error(f"Error starting WebSocket for {exchange_name}: {str(e)}")
    
    async def stop_websocket_feeds(self):
        """Stop all WebSocket feeds."""
        if not self.ws_running:
            return
        
        self.ws_running = False
        
        for exchange_name, manager in self.ws_managers.items():
            if 'task' in manager:
                manager['task'].cancel()
                try:
                    await manager['task']
                except asyncio.CancelledError:
                    pass
                logger.info(f"Stopped WebSocket for {exchange_name}")
        
        self.ws_managers = {}
    
    async def _websocket_manager(self, exchange_name: str, url: str):
        """
        WebSocket connection manager for an exchange.
        
        Args:
            exchange_name: Name of the exchange
            url: WebSocket URL
        """
        backoff = 1  # Initial backoff in seconds
        max_backoff = 60  # Maximum backoff in seconds
        
        while self.ws_running:
            try:
                # Check circuit breaker
                if self.circuit_breakers[exchange_name]['open']:
                    # Check if we should reset the circuit breaker
                    if time.time() - self.circuit_breakers[exchange_name]['last_failure'] > 300:  # 5 minutes
                        self.circuit_breakers[exchange_name]['open'] = False
                        self.circuit_breakers[exchange_name]['failures'] = 0
                        logger.info(f"Circuit breaker reset for {exchange_name}")
                    else:
                        # Circuit breaker is open, wait before retrying
                        await asyncio.sleep(10)
                        continue
                
                # Connect to WebSocket
                async with websockets.connect(url) as websocket:
                    logger.info(f"Connected to {exchange_name} WebSocket")
                    
                    # Subscribe to relevant channels
                    subscribe_msg = self.exchange_wrappers[exchange_name].get_websocket_subscribe_message()
                    await websocket.send(subscribe_msg)
                    
                    # Reset backoff on successful connection
                    backoff = 1
                    
                    # Process messages
                    while self.ws_running:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                            await self._process_websocket_message(exchange_name, message)
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            try:
                                pong = await websocket.ping()
                                await asyncio.wait_for(pong, timeout=10)
                            except:
                                # Ping failed, break to reconnect
                                logger.warning(f"{exchange_name} WebSocket ping failed, reconnecting...")
                                break
                        except Exception as e:
                            logger.error(f"Error processing {exchange_name} WebSocket message: {str(e)}")
                            # Don't break on processing errors
            
            except Exception as e:
                # Update circuit breaker
                self.circuit_breakers[exchange_name]['failures'] += 1
                self.circuit_breakers[exchange_name]['last_failure'] = time.time()
                
                # Open circuit breaker if too many failures
                if self.circuit_breakers[exchange_name]['failures'] >= 5:
                    self.circuit_breakers[exchange_name]['open'] = True
                    logger.warning(f"Circuit breaker opened for {exchange_name}")
                
                logger.error(f"WebSocket error for {exchange_name}: {str(e)}")
                
                # Exponential backoff
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
    
    async def _process_websocket_message(self, exchange_name: str, message: str):
        """
        Process a WebSocket message from an exchange.
        
        Args:
            exchange_name: Name of the exchange
            message: WebSocket message
        """
        try:
            # Parse the message
            parsed = self.exchange_wrappers[exchange_name].parse_websocket_message(message)
            
            if not parsed:
                return
            
            # Update the graph with new data
            async with self.lock:
                updated = False
                
                for update in parsed:
                    if 'type' not in update:
                        continue
                    
                    if update['type'] == 'orderbook':
                        # Extract trading pair and rates
                        symbol = update.get('symbol')
                        if not symbol or '/' not in symbol:
                            continue
                        
                        base, quote = symbol.split('/')
                        
                        # Get best bid and ask
                        if 'bids' in update and update['bids'] and 'asks' in update and update['asks']:
                            best_bid = float(update['bids'][0][0])
                            best_ask = float(update['asks'][0][0])
                            
                            # Calculate liquidity
                            bid_liquidity = sum(float(qty) for price, qty in update['bids'][:5])
                            ask_liquidity = sum(float(qty) for price, qty in update['asks'][:5])
                            
                            # Apply depth threshold pruning
                            if (bid_liquidity < self.config['depth_threshold'] or 
                                ask_liquidity < self.config['depth_threshold']):
                                continue
                            
                            # Add edges to the graph
                            # Quote to base (buy)
                            updated |= self.graph.add_edge(
                                from_asset=quote,
                                to_asset=base,
                                exchange=exchange_name,
                                rate=1.0 / best_ask,
                                liquidity=ask_liquidity
                            )
                            
                            # Base to quote (sell)
                            updated |= self.graph.add_edge(
                                from_asset=base,
                                to_asset=quote,
                                exchange=exchange_name,
                                rate=best_bid,
                                liquidity=bid_liquidity
                            )
                
                # If the graph was updated, scan for arbitrage opportunities
                if updated and self.config['incremental']:
                    # Get subgraph for incremental processing
                    subgraph = self.graph.get_subgraph(
                        self.graph.dirty_nodes,
                        radius=self.config['subgraph_radius']
                    )
                    
                    # Scan for arbitrage in the subgraph
                    await self._scan_for_arbitrage_incremental(subgraph)
                    
                    # Clear dirty flags
                    self.graph.clear_dirty_flags()
        
        except Exception as e:
            logger.error(f"Error processing {exchange_name} WebSocket message: {str(e)}")
    
    async def _scan_for_arbitrage_incremental(self, subgraph: Set[str]):
        """
        Scan for arbitrage opportunities in a subgraph.
        
        Args:
            subgraph: Set of nodes to include in the scan
        """
        # Choose algorithm based on configuration
        if self.config['algorithm'] == 'johnson':
            opportunities = JohnsonAlgorithm.find_negative_cycles(self.graph, subgraph)
        elif self.config['algorithm'] == 'bellman-ford':
            opportunities = BellmanFordAlgorithm.find_negative_cycles(self.graph, subgraph)
        elif self.config['algorithm'] == 'yen-k-paths':
            # For Yen's algorithm, we need a source and target
            # Use USDT as default if available
            if "USDT" in subgraph:
                source = "USDT"
                # Find a target that's not the source
                targets = [node for node in subgraph if node != source]
                if targets:
                    target = targets[0]
                    opportunities = YenKShortestPaths.find_k_shortest_paths(
                        self.graph, source, target, self.config['k_paths']
                    )
                else:
                    opportunities = []
            else:
                opportunities = []
        else:
            logger.warning(f"Unknown algorithm: {self.config['algorithm']}, falling back to Johnson's")
            opportunities = JohnsonAlgorithm.find_negative_cycles(self.graph, subgraph)
        
        # Process opportunities
        for opportunity in opportunities:
            # Add to cache
            self.add_loop_to_cache(opportunity['loop'], {
                'profit_percent': opportunity.get('profit_percent', 0),
                'exchanges': opportunity.get('exchanges', [])
            })
    
    def scan_for_opportunities(self, loops: List[List[str]] = None) -> List[Dict]:
        """
        Scan for arbitrage opportunities across all exchanges and loops.
        
        Args:
            loops: Optional list of trading loops to check, if None uses cached loops
            
        Returns:
            List of profitable opportunities
        """
        # If no loops provided, use cached loops
        if loops is None:
            loops = [entry["pairs"] for entry in self.loop_cache["loops"]]
            
            # If still no loops, use some default triangular loops
            if not loops:
                loops = [
                    ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
                    ["ETH/USDT", "XRP/ETH", "XRP/USDT"],
                    ["BTC/USDT", "LTC/BTC", "LTC/USDT"]
                ]
        
        opportunities = []
        
        if self.config['parallel_scanning']:
            # Parallel scanning for better performance
            with ThreadPoolExecutor(max_workers=min(10, len(self.exchange_wrappers) * len(loops))) as executor:
                futures = []
                for exchange_name, wrapper in self.exchange_wrappers.items():
                    # Skip exchanges with open circuit breakers
                    if self.circuit_breakers[exchange_name]['open']:
                        logger.info(f"Skipping {exchange_name} due to open circuit breaker")
                        continue
                        
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
                # Skip exchanges with open circuit breakers
                if self.circuit_breakers[exchange_name]['open']:
                    logger.info(f"Skipping {exchange_name} due to open circuit breaker")
                    continue
                    
                for loop in loops:
                    result = self.check_loop_profitability(exchange_name, wrapper, loop)
                    if result:
                        opportunities.append(result)
        
        # Sort by risk-adjusted profit
        return sorted(
            opportunities, 
            key=lambda x: x.get('risk_adjusted_profit', 0), 
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
        start_time = time.time()
        
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
            
            # Calculate execution duration
            duration = int((time.time() - start_time) * 1000)
            
            # Check if profitable after all adjustments
            if adjusted_profit_pct > self.config['min_profit_threshold']:
                # Calculate risk metrics
                volatility = self._estimate_price_volatility(loop, wrapper)
                liquidity_risk = self._calculate_liquidity_risk(ob1, ob2, ob3, position_size)
                execution_risk = min(1.0, duration / self.config['max_execution_time'])
                
                # Combined risk score (0-1 scale, lower is better)
                risk_score = (volatility * 0.4) + (liquidity_risk * 0.4) + (execution_risk * 0.2)
                
                # Risk-adjusted profit
                risk_adjusted_profit = adjusted_profit_pct * (1 - risk_score)
                
                # Calculate risk-reward ratio
                risk_reward_ratio = adjusted_profit_pct / (risk_score * 100) if risk_score > 0 else float('inf')
                
                # Only return if risk-reward ratio is acceptable
                if risk_reward_ratio >= self.config['risk_reward_ratio']:
                    return {
                        'exchange': exchange_name,
                        'exchanges': [exchange_name, exchange_name, exchange_name],  # Same exchange for all steps
                        'loop': loop,
                        'position_size': position_size,
                        'raw_profit_percent': profit_pct,
                        'adjusted_profit_percent': adjusted_profit_pct,
                        'risk_score': risk_score,
                        'risk_adjusted_profit': risk_adjusted_profit,
                        'risk_reward_ratio': risk_reward_ratio,
                        'execution_time_ms': duration,
                        'timestamp': time.time(),
                        'trade_simulation': {
                            'first_trade': first_trade_result,
                            'second_trade': second_trade_result,
                            'final_trade': final_trade_result
                        }
                    }
        
        except Exception as e:
            # Update circuit breaker
            self.circuit_breakers[exchange_name]['failures'] += 1
            self.circuit_breakers[exchange_name]['last_failure'] = time.time()
            
            # Open circuit breaker if too many failures
            if self.circuit_breakers[exchange_name]['failures'] >= 5:
                self.circuit_breakers[exchange_name]['open'] = True
                logger.warning(f"Circuit breaker opened for {exchange_name}")
            
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
            if len(ob['bids']) < 3 or len(ob['asks']) < 3:  # Require minimum depth
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
    
    def _estimate_price_volatility(self, loop: List[str], wrapper: Any) -> float:
        """
        Estimate price volatility for risk assessment.
        
        Returns a value between 0-1 where higher means more volatile.
        """
        try:
            # Try to get recent price changes if available
            # This is a placeholder - actual implementation would depend on the wrapper
            # For now, return a moderate volatility estimate
            return 0.5
        except:
            # Fallback to a moderate volatility estimate
            return 0.5
    
    def _calculate_liquidity_risk(self, ob1: Dict, ob2: Dict, ob3: Dict, size: float) -> float:
        """
        Calculate liquidity risk based on order book depth.
        
        Returns a value between 0-1 where higher means more risk.
        """
        try:
            # Calculate bid-ask spreads as percentage
            spread1 = (float(ob1['asks'][0][0]) / float(ob1['bids'][0][0]) - 1) * 100
            spread2 = (float(ob2['asks'][0][0]) / float(ob2['bids'][0][0]) - 1) * 100
            spread3 = (float(ob3['asks'][0][0]) / float(ob3['bids'][0][0]) - 1) * 100
            
            # Average spread as risk indicator
            avg_spread = (spread1 + spread2 + spread3) / 3
            
            # Calculate order book depth ratio (how much of the order book we're using)
            depth_usage = size / self._calculate_optimal_size(ob1, ob2, ob3)
            
            # Combine factors (higher spread and higher depth usage = higher risk)
            liquidity_risk = min(1.0, (avg_spread / 5) * 0.5 + depth_usage * 0.5)
            
            return liquidity_risk
        except Exception as e:
            # Fallback to moderate risk if calculation fails
            logger.error(f"Error calculating liquidity risk: {str(e)}")
            return 0.5
    
    def execute_arbitrage(self, opportunity: Dict) -> Dict:
        """
        Execute an arbitrage opportunity.
        
        Args:
            opportunity: Opportunity details from check_loop_profitability
            
        Returns:
            Execution results
        """
        exchange_name = opportunity['exchange']
        loop = opportunity['loop']
        position_size = opportunity['position_size']
        
        try:
            # Check circuit breaker
            if self.circuit_breakers[exchange_name]['open']:
                return {
                    'success': False,
                    'error': f"Circuit breaker open for {exchange_name}"
                }
            
            wrapper = self.exchange_wrappers[exchange_name]
            
            # Record start time
            start_time = time.time()
            
            # Execute first trade (buy)
            first_trade = wrapper.create_market_order(
                symbol=loop[0],
                side='buy',
                amount=position_size
            )
            
            # Get resulting amount
            first_amount = first_trade['filled']
            
            # Execute second trade
            second_trade = wrapper.create_market_order(
                symbol=loop[1],
                side='sell',
                amount=first_amount
            )
            
            # Get resulting amount
            second_amount = second_trade['filled']
            
            # Execute final trade
            final_trade = wrapper.create_market_order(
                symbol=loop[2],
                side='sell',
                amount=second_amount
            )
            
            # Calculate actual profit
            final_amount = final_trade['cost']
            actual_profit = final_amount - position_size
            actual_profit_pct = (actual_profit / position_size) * 100
            
            # Calculate execution time
            execution_time = int((time.time() - start_time) * 1000)
            
            # Record trade in history
            trade_record = {
                'timestamp': time.time(),
                'exchange': exchange_name,
                'loop': loop,
                'position_size': position_size,
                'expected_profit_percent': opportunity['adjusted_profit_percent'],
                'actual_profit_percent': actual_profit_pct,
                'execution_time_ms': execution_time,
                'success': True
            }
            
            self.trade_history.append(trade_record)
            self._update_performance_metrics()
            
            # Reset circuit breaker failures on success
            self.circuit_breakers[exchange_name]['failures'] = 0
            
            return {
                'success': True,
                'profit_percent': actual_profit_pct,
                'execution_time_ms': execution_time,
                'trades': [first_trade, second_trade, final_trade]
            }
            
        except Exception as e:
            # Update circuit breaker
            self.circuit_breakers[exchange_name]['failures'] += 1
            self.circuit_breakers[exchange_name]['last_failure'] = time.time()
            
            # Open circuit breaker if too many failures
            if self.circuit_breakers[exchange_name]['failures'] >= 5:
                self.circuit_breakers[exchange_name]['open'] = True
                logger.warning(f"Circuit breaker opened for {exchange_name}")
            
            # Record failed trade
            trade_record = {
                'timestamp': time.time(),
                'exchange': exchange_name,
                'loop': loop,
                'position_size': position_size,
                'expected_profit_percent': opportunity['adjusted_profit_percent'],
                'actual_profit_percent': 0,
                'execution_time_ms': int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
                'success': False,
                'error': str(e)
            }
            
            self.trade_history.append(trade_record)
            self._update_performance_metrics()
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_performance_metrics(self):
        """Update performance metrics based on trade history."""
        if not self.trade_history:
            return
            
        # Calculate total profit
        total_profit = sum(
            trade['actual_profit_percent'] * trade['position_size'] / 100 
            for trade in self.trade_history if trade['success']
        )
        
        # Calculate win rate
        successful_trades = sum(1 for trade in self.trade_history if trade['success'] and trade['actual_profit_percent'] > 0)
        win_rate = successful_trades / len(self.trade_history) if self.trade_history else 0
        
        # Calculate average profit
        avg_profit = sum(
            trade['actual_profit_percent'] for trade in self.trade_history if trade['success']
        ) / successful_trades if successful_trades else 0
        
        # Calculate max drawdown
        # (simplified implementation)
        equity_curve = []
        equity = 1000  # Starting equity
        for trade in sorted(self.trade_history, key=lambda x: x['timestamp']):
            if trade['success']:
                profit = trade['position_size'] * trade['actual_profit_percent'] / 100
                equity += profit
            equity_curve.append(equity)
            
        max_drawdown = 0
        peak = equity_curve[0] if equity_curve else 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Update metrics
        self.performance_metrics = {
            'total_profit': total_profit,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_drawdown': max_drawdown,
            'trade_count': len(self.trade_history)
        }
        
    def get_performance_report(self) -> Dict:
        """Get performance report with metrics and recommendations."""
        self._update_performance_metrics()
        
        return {
            'metrics': self.performance_metrics,
            'recent_trades': self.trade_history[-10:] if self.trade_history else [],
            'recommendations': self._generate_recommendations(),
            'circuit_breakers': {k: v['open'] for k, v in self.circuit_breakers.items()}
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on performance."""
        recommendations = []
        
        if not self.trade_history or len(self.trade_history) < 5:
            recommendations.append("Insufficient trade history for meaningful recommendations.")
            return recommendations
            
        # Win rate recommendations
        if self.performance_metrics['win_rate'] < 0.5:
            recommendations.append(
                "Low win rate detected. Consider increasing the minimum profit threshold or safety margin."
            )
        
        # Profit recommendations
        if self.performance_metrics['avg_profit'] < self.config['min_profit_threshold']:
            recommendations.append(
                "Average profit below threshold. Consider focusing on higher quality opportunities."
            )
            
        # Drawdown recommendations
        if self.performance_metrics['max_drawdown'] > 0.1:
            recommendations.append(
                "Significant drawdown detected. Consider reducing position sizes or implementing stricter risk controls."
            )
            
        # Analyze failed trades
        failed_trades = [t for t in self.trade_history if not t['success']]
        if failed_trades:
            common_errors = {}
            for trade in failed_trades:
                error = trade.get('error', 'Unknown error')
                common_errors[error] = common_errors.get(error, 0) + 1
                
            most_common_error = max(common_errors.items(), key=lambda x: x[1]) if common_errors else None
            if most_common_error:
                recommendations.append(
                    f"Most common error: {most_common_error[0]}. Consider addressing this issue."
                )
        
        # Circuit breaker recommendations
        open_breakers = [k for k, v in self.circuit_breakers.items() if v['open']]
        if open_breakers:
            recommendations.append(
                f"Circuit breakers open for: {', '.join(open_breakers)}. Check exchange connectivity."
            )
        
        return recommendations
    
    def start(self):
        """Start the scanner with WebSocket feeds."""
        if self.event_loop_thread is not None:
            return
        
        def run_event_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.start_websocket_feeds())
            self.loop.run_forever()
        
        self.event_loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self.event_loop_thread.start()
        logger.info("Started WebSocket feeds")
    
    def stop(self):
        """Stop the scanner and WebSocket feeds."""
        if self.event_loop_thread is None:
            return
        
        async def shutdown():
            await self.stop_websocket_feeds()
            self.loop.stop()
        
        asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
        self.event_loop_thread = None
        logger.info("Stopped WebSocket feeds")
