import json,os
from typing import Dict
from core.config import get_data_path

# Use absolute path from config
BALANCE_FILE = get_data_path('paper_balance.json')

class PaperTrader:
    def __init__(self):
        """
        Initialize the PaperTrader with a balance from file or default.
        """
        self.balance = self.load_balance()
        
    def load_balance(self):
        """
        Load balance from file or create with default if not exists.
        
        Returns:
            dict: Current balance
        """
        try:
            if os.path.exists(BALANCE_FILE):
                with open(BALANCE_FILE, 'r') as f:
                    return json.load(f)
            else:
                # Create default balance
                default_balance = {'USDT': 1000}
                self.save_balance(default_balance)
                return default_balance
        except Exception as e:
            print(f"Error loading balance: {str(e)}")
            return {'USDT': 1000}  # Fallback to default
            
    def save_balance(self, balance):
        """
        Save balance to file.
        
        Args:
            balance (dict): Balance to save
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(BALANCE_FILE), exist_ok=True)
            with open(BALANCE_FILE, 'w') as f:
                json.dump(balance, f, indent=2)
        except Exception as e:
            print(f"Error saving balance: {str(e)}")
    
    def execute_trade(self, trade: Dict):
        """
        Execute a paper trade and update balance.
        
        Args:
            trade (dict): Trade information including profit percentage
        """
        try:
            # Calculate profit based on trade information
            profit = trade.get('profit_percent', 0) / 100 * 100
            
            # Update balance
            self.balance['USDT'] += profit
            
            # Save updated balance
            self.save_balance(self.balance)
            
            # Return updated balance for confirmation
            return {
                'success': True,
                'profit': profit,
                'new_balance': self.balance
            }
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
