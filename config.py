import os
import json
from typing import Dict, Optional

def load_api_keys():
    """
    Load API keys from environment variables first, then fallback to file.
    Returns a dictionary of exchange names to API credentials.
    """
    # Try to load from environment variables first
    api_keys = {}
    exchanges = ["gateio", "mexc", "phemex", "bitget"]
    
    for ex in exchanges:
        key = os.environ.get(f"{ex.upper()}_API_KEY")
        secret = os.environ.get(f"{ex.upper()}_API_SECRET")
        
        if key and secret:
            api_keys[ex] = {"api_key": key, "api_secret": secret}
    
    # Fall back to file if environment variables not set
    if not api_keys and os.path.exists("api_keys.json"):
        with open("api_keys.json", "r") as f:
            file_keys = json.load(f)
            if file_keys:
                api_keys = file_keys
    
    return api_keys

def get_base_dir():
    """Get the absolute path to the base directory of the application."""
    return os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename):
    """Get the absolute path to a data file."""
    data_dir = os.path.join(get_base_dir(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)

def get_default_algorithm_config():
    """Get default configuration for the optimized algorithm."""
    return {
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
    }

def save_algorithm_config(config: Dict):
    """Save algorithm configuration to file."""
    config_path = get_data_path("algorithm_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def load_algorithm_config() -> Dict:
    """Load algorithm configuration from file or return defaults."""
    config_path = get_data_path("algorithm_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except:
            pass
    return get_default_algorithm_config()
