# Cryptocurrency Arbitrage Bot - Ultimate Edition

## Overview

This advanced cryptocurrency arbitrage bot is designed to identify and execute profitable trading opportunities across multiple exchanges. It features state-of-the-art algorithms, machine learning capabilities, real-time WebSocket data feeds, and a comprehensive dashboard for monitoring and control.

## Key Features

- **Multiple Algorithm Options**:
  - Basic: Simple triangular arbitrage
  - Optimized: Fee and slippage-aware calculations
  - Advanced: Graph theory-based cycle detection with WebSockets
  - ML-Enhanced: Machine learning scoring and risk assessment

- **Real-Time Data Processing**:
  - WebSocket integration for instant market updates
  - Parallel scanning across multiple exchanges
  - Incremental graph updates for efficient processing

- **Risk Management**:
  - Dynamic position sizing based on liquidity
  - Slippage modeling using order book depth
  - Fee-aware profit calculation
  - Risk scoring and circuit breakers

- **Advanced Trading Features**:
  - Auto-trading with configurable parameters
  - Performance tracking and analytics
  - Trade simulation with detailed breakdowns
  - Paper and live trading modes

- **User-Friendly Dashboard**:
  - Real-time opportunity monitoring
  - Performance metrics and charts
  - Advanced configuration options
  - Trading loop management

## Supported Exchanges

- GateIO
- MEXC
- Phemex
- Bitget

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository or extract the zip file:
   ```
   git clone https://github.com/yourusername/arbitrage-bot.git
   cd arbitrage-bot
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Edit `api_keys.json` with your exchange API keys
   - Alternatively, set environment variables (see Configuration section)

## Usage

### Starting the Bot

Run the dashboard interface:
```
streamlit run dashboard.py
```

This will open a web interface in your browser where you can control all aspects of the bot.

### Dashboard Controls

- **Algorithm Selection**: Choose between Basic, Optimized, Advanced, or ML-Enhanced algorithms
- **Exchange Selection**: Select which exchanges to include in scanning
- **Trading Mode**: Switch between Paper trading (simulation) and Live trading
- **Scanner Controls**: Start/stop the scanner and WebSocket feeds
- **Trading Controls**: Enable/disable auto-trading and pause/resume trading
- **Advanced Settings**: Configure algorithm parameters, risk thresholds, and more

### Configuration

#### API Keys

API keys can be configured in two ways:

1. Edit the `api_keys.json` file:
   ```json
   {
     "gateio": {
       "api_key": "YOUR_API_KEY",
       "api_secret": "YOUR_API_SECRET"
     },
     "mexc": {
       "api_key": "YOUR_API_KEY",
       "api_secret": "YOUR_API_SECRET"
     },
     "phemex": {
       "api_key": "YOUR_API_KEY",
       "api_secret": "YOUR_API_SECRET"
     },
     "bitget": {
       "api_key": "YOUR_API_KEY",
       "api_secret": "YOUR_API_SECRET"
     }
   }
   ```

2. Set environment variables:
   ```
   GATEIO_API_KEY=your_api_key
   GATEIO_API_SECRET=your_api_secret
   MEXC_API_KEY=your_api_key
   MEXC_API_SECRET=your_api_secret
   PHEMEX_API_KEY=your_api_key
   PHEMEX_API_SECRET=your_api_secret
   BITGET_API_KEY=your_api_key
   BITGET_API_SECRET=your_api_secret
   ```

#### Algorithm Configuration

Algorithm settings can be configured through the dashboard interface or by editing `data/algorithm_config.json`:

```json
{
  "min_profit_threshold": 0.5,
  "safety_margin": 0.2,
  "max_slippage": 0.3,
  "order_book_depth": 15,
  "max_position_size": 1000,
  "min_position_size": 10,
  "parallel_scanning": true,
  "algorithm": "johnson",
  "incremental": true,
  "subgraph_radius": 2,
  "ml_scoring": true,
  "reinforcement_learning": true,
  "depth_threshold": 0.1,
  "volatility_threshold": 0.5
}
```

## Free 24/7 Deployment Options

### Option 1: Render (Recommended)

1. Create a free account on [Render](https://render.com/)
2. Create a new Web Service
3. Connect your GitHub repository or upload the code
4. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run dashboard.py`
   - Environment Variables: Add your API keys
5. Deploy the service

### Option 2: Streamlit Community Cloud

1. Create a free account on [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Configure the deployment:
   - Main file path: `dashboard.py`
   - Python version: 3.9+
   - Requirements: `requirements.txt`
   - Secrets: Add your API keys
4. Deploy the app

### Option 3: PythonAnywhere

1. Create a free account on [PythonAnywhere](https://www.pythonanywhere.com/)
2. Upload your code or clone from GitHub
3. Set up a web app:
   - Framework: Flask
   - Python version: 3.9+
4. Configure a scheduled task to run the bot
5. Set environment variables for API keys

### Option 4: Heroku (Low-cost option)

1. Create an account on [Heroku](https://www.heroku.com/)
2. Install the Heroku CLI
3. Create a new app:
   ```
   heroku create arbitrage-bot
   ```
4. Configure the app:
   - Create a `Procfile` with: `web: streamlit run dashboard.py`
   - Set environment variables for API keys
5. Deploy the app:
   ```
   git push heroku main
   ```

## Advanced Algorithm Details

### Basic Algorithm

The basic algorithm performs simple triangular arbitrage by:
1. Checking price differences across three trading pairs forming a loop
2. Using only the top bid/ask prices from order books
3. Calculating raw profit percentage without fees or slippage

### Optimized Algorithm

The optimized algorithm enhances the basic approach by:
1. Incorporating exchange fees in all calculations
2. Modeling slippage using order book depth
3. Calculating optimal position size based on liquidity
4. Applying safety margins for execution latency

### Advanced Algorithm

The advanced algorithm uses graph theory for arbitrage detection:
1. Represents trading pairs as a weighted graph
2. Uses Johnson's algorithm to find negative cycles (arbitrage opportunities)
3. Processes real-time updates via WebSockets
4. Implements incremental updates for efficiency
5. Includes circuit breakers for exchange reliability

### ML-Enhanced Algorithm

The ML-enhanced algorithm adds machine learning capabilities:
1. Scores opportunities based on multiple factors
2. Uses reinforcement learning to improve over time
3. Adapts to changing market conditions
4. Provides risk-adjusted profit metrics
5. Generates recommendations based on performance

## Project Structure

```
arbitrage_bot/
├── core/
│   ├── advanced_scanner.py     # Advanced graph-based scanner
│   ├── config.py               # Configuration utilities
│   ├── executor.py             # Trade execution logic
│   ├── graph_algorithms.py     # Graph theory algorithms
│   ├── loop_manager.py         # Trading loop management
│   ├── ml_scanner.py           # ML-enhanced scanner
│   ├── optimized_scanner.py    # Fee and slippage aware scanner
│   └── scanner.py              # Basic scanner
├── wrappers/
│   ├── gateio_wrapper.py       # GateIO exchange wrapper
│   ├── mexc_wrapper.py         # MEXC exchange wrapper
│   ├── phemex_wrapper.py       # Phemex exchange wrapper
│   └── bitget_wrapper.py       # Bitget exchange wrapper
├── data/
│   ├── cached_loops.json       # Cached trading loops
│   ├── algorithm_config.json   # Algorithm configuration
│   └── trade_history.json      # Trade execution history
├── logs/                       # Log files directory
├── api_keys.json               # API key configuration
├── dashboard.py                # Streamlit dashboard interface
├── main.py                     # Command-line interface
└── requirements.txt            # Python dependencies
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failures**
   - Check your internet connection
   - Verify that the exchange is operational
   - Try restarting the WebSocket feeds

2. **No Profitable Opportunities**
   - Try decreasing the minimum profit threshold
   - Add more exchanges or trading pairs
   - Check if market conditions are suitable for arbitrage

3. **API Key Errors**
   - Verify that your API keys are correct
   - Ensure you have enabled trading permissions
   - Check if IP restrictions are in place

4. **High Execution Time**
   - Enable parallel scanning
   - Use the Advanced or ML-Enhanced algorithm
   - Reduce the number of trading loops

### Logs

Log files are stored in the `logs` directory and can be useful for diagnosing issues.

## Performance Optimization

For optimal performance:

1. Use the Advanced or ML-Enhanced algorithm for most efficient scanning
2. Enable WebSocket feeds for real-time updates
3. Use parallel scanning for faster processing
4. Configure appropriate position sizes based on your capital
5. Adjust risk parameters based on market conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and can result in the loss of your invested capital. The authors are not responsible for any financial losses incurred while using this software.
