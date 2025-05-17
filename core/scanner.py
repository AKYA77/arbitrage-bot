from typing import Dict, Any
import time

class Scanner:
    def __init__(self, wrapper):
        """
        Initialize the Scanner with an exchange wrapper.
        
        Args:
            wrapper: Exchange wrapper instance with get_order_book method
        """
        self.wrapper = wrapper
        
    def check_profitability(self, loop):
        """
        Check the profitability of a trading loop.
        
        Args:
            loop (list): List of three trading pairs forming a loop
            
        Returns:
            dict: Profitability information including profit percentage and duration
            
        Example:
            loop = ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
            result = scanner.check_profitability(loop)
        """
        symbols = loop
        amt = 100  # Starting amount
        start_time = time.time()
        
        try:
            # Validate loop format
            if len(symbols) != 3:
                raise ValueError(f"Invalid loop format: {symbols}. Must contain exactly 3 trading pairs.")
            
            # Get order books for all pairs in the loop
            ob1 = self.wrapper.get_order_book(symbols[0])
            ob2 = self.wrapper.get_order_book(symbols[1])
            ob3 = self.wrapper.get_order_book(symbols[2])
            
            # Validate order book data
            for i, ob in enumerate([ob1, ob2, ob3]):
                if not ob or 'bids' not in ob or 'asks' not in ob:
                    raise ValueError(f"Invalid order book data for {symbols[i]}")
                if not ob['bids'] or not ob['asks']:
                    raise ValueError(f"Empty order book for {symbols[i]}")
            
            # Calculate profitability
            # First trade: Convert base currency to first asset
            x = amt / ob1['asks'][0][0]
            
            # Second trade: Convert first asset to second asset
            y = x * ob2['bids'][0][0]
            
            # Third trade: Convert second asset back to base currency
            final = y * ob3['bids'][0][0]
            
            # Calculate profit percentage
            pct = (final - amt) / amt * 100
            
            # Calculate duration in milliseconds
            duration = int((time.time() - start_time) * 1000)
            
            return {
                'profit_percent': round(pct, 4),
                'duration': duration,
                'start_amount': amt,
                'final_amount': round(final, 4),
                'trades': [
                    f"{amt} → {round(x, 6)} via {symbols[0]}",
                    f"{round(x, 6)} → {round(y, 6)} via {symbols[1]}",
                    f"{round(y, 6)} → {round(final, 4)} via {symbols[2]}"
                ]
            }
            
        except Exception as e:
            # Log the error but return empty dict to avoid crashing
            print(f"Error checking profitability for loop {symbols}: {str(e)}")
            return {}
        finally:
            # Ensure duration is calculated even if there's an error
            if 'start_time' in locals():
                duration = int((time.time() - start_time) * 1000)
