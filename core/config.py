import os
import json
from typing import Dict, Any, Optional

def load_api_keys() -> Dict[str, Dict[str, str]]:
    """
    Load API keys from the api_keys.json file or environment variables.
    
    Returns:
        Dict of exchange name to API key information
    """
    # First try to load from environment variables
    api_keys = {}
    
    # Check for environment variables (format: EXCHANGE_API_KEY, EXCHANGE_API_SECRET)
    exchanges = ["gateio", "mexc", "phemex", "bitget"]
    for exchange in exchanges:
        api_key_env = f"{exchange.upper()}_API_KEY"
        api_secret_env = f"{exchange.upper()}_API_SECRET"
        
        if api_key_env in os.environ and api_secret_env in os.environ:
            api_keys[exchange] = {
                "api_key": os.environ[api_key_env],
                "api_secret": os.environ[api_secret_env]
            }
    
    # If no keys found in environment, try loading from file
    if not api_keys:
        try:
            api_keys_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_keys.json")
            if os.path.exists(api_keys_path):
                with open(api_keys_path, 'r') as f:
                    api_keys = json.load(f)
        except Exception as e:
            print(f"Error loading API keys: {str(e)}")
            api_keys = {}
    
    return api_keys

def get_default_algorithm_config() -> Dict[str, Any]:
    """
    Get default algorithm configuration.
    
    Returns:
        Dict of configuration parameters
    """
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
        'algorithm': 'johnson',       # Algorithm to use: 'johnson', 'bellman-ford', 'yen-k-paths'
        'k_paths': 5,                 # Number of paths to find with Yen's algorithm
        'incremental': True,          # Whether to use incremental updates
        'subgraph_radius': 2,         # Radius for subgraph extraction in incremental updates
        'ml_scoring': True,           # Whether to use ML for loop scoring
        'reinforcement_learning': True, # Whether to use RL for loop prioritization
        'depth_threshold': 0.1,       # Minimum depth threshold for pruning
        'volatility_threshold': 0.5,  # Maximum volatility for considering an opportunity
    }

def save_algorithm_config(config: Dict[str, Any]) -> None:
    """
    Save algorithm configuration to file.
    
    Args:
        config: Configuration parameters
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "algorithm_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving algorithm configuration: {str(e)}")

def load_algorithm_config() -> Dict[str, Any]:
    """
    Load algorithm configuration from file.
    
    Returns:
        Dict of configuration parameters
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "algorithm_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading algorithm configuration: {str(e)}")
    
    # Return default config if loading fails
    return get_default_algorithm_config()

def get_data_dir() -> str:
    """
    Get the data directory path, creating it if it doesn't exist.
    
    Returns:
        Absolute path to the data directory
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_logs_dir() -> str:
    """
    Get the logs directory path, creating it if it doesn't exist.
    
    Returns:
        Absolute path to the logs directory
    """
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def get_file_path(filename: str, directory: Optional[str] = None) -> str:
    """
    Get the absolute path for a file in a specific directory.
    
    Args:
        filename: Name of the file
        directory: Optional directory name (relative to project root)
        
    Returns:
        Absolute path to the file
    """
    if directory:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, filename)
    else:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)
