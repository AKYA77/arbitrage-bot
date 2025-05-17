import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta

from core.loop_manager import LoopManager
from core.advanced_scanner import AdvancedArbitrageScanner
from core.ml_scanner import MLArbitrageScanner
from core.config import load_api_keys, get_default_algorithm_config, save_algorithm_config, load_algorithm_config

from wrappers.gateio_wrapper import GateIOWrapper
from wrappers.mexc_wrapper import MEXCWrapper
from wrappers.phemex_wrapper import PhemexWrapper
from wrappers.bitget_wrapper import BitgetWrapper

# Set page config
st.set_page_config(
    page_title="Advanced Arbitrage Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .profit-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .profit-negative {
        color: #F44336;
        font-weight: bold;
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
    }
    .warning-text {
        color: #FFC107;
        font-weight: bold;
    }
    .error-text {
        color: #F44336;
        font-weight: bold;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.scanner_running = False
    st.session_state.websocket_running = False
    st.session_state.scanner_instance = None
    st.session_state.last_scan_time = None
    st.session_state.scan_results = []
    st.session_state.performance_metrics = {}
    st.session_state.trade_history = []
    st.session_state.active_exchanges = ["gateio", "mexc", "phemex", "bitget"]
    st.session_state.algorithm = "advanced"
    st.session_state.trading_mode = "Paper"
    st.session_state.trading_paused = False
    st.session_state.auto_trading = False
    st.session_state.scan_interval = 30
    st.session_state.min_profit_threshold = 0.5
    st.session_state.safety_margin = 0.2
    st.session_state.max_position_size = 1000
    st.session_state.parallel_scanning = True
    st.session_state.show_advanced = False
    st.session_state.show_profit = True
    st.session_state.show_exchange = True
    st.session_state.show_duration = True
    st.session_state.show_risk = True
    st.session_state.show_risk_adjusted = True
    st.session_state.timeframe = "30 mins"
    st.session_state.trigger_scan = False
    st.session_state.initialized = True

# Header
st.markdown("<h1 class='main-header'>Advanced Crypto Arbitrage Bot</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Algorithm Selection")
    algorithm_options = {
        "basic": "Basic (Simple Triangular)",
        "optimized": "Optimized (Fee & Slippage Aware)",
        "advanced": "Advanced (WebSocket & Graph Theory)",
        "ml": "ML-Enhanced (Machine Learning)"
    }
    selected_algorithm = st.selectbox(
        "Select Algorithm",
        options=list(algorithm_options.keys()),
        format_func=lambda x: algorithm_options[x],
        index=list(algorithm_options.keys()).index(st.session_state.algorithm)
    )
    st.session_state.algorithm = selected_algorithm
    
    st.markdown("### Exchange Selection")
    exchanges = ["gateio", "mexc", "phemex", "bitget"]
    active_exchanges = []
    for ex in exchanges:
        is_active = st.checkbox(ex.upper(), value=ex in st.session_state.active_exchanges)
        if is_active:
            active_exchanges.append(ex)
    st.session_state.active_exchanges = active_exchanges
    
    st.markdown("### Trading Mode")
    trading_mode = st.selectbox(
        "Trading Mode",
        options=["Paper", "Live"],
        index=0 if st.session_state.trading_mode == "Paper" else 1
    )
    st.session_state.trading_mode = trading_mode
    
    if st.session_state.trading_mode == "Live":
        st.warning("⚠️ Live trading mode will execute real trades using your API keys!")
    
    st.markdown("### Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Scanner" if not st.session_state.scanner_running else "Stop Scanner"):
            st.session_state.scanner_running = not st.session_state.scanner_running
            if st.session_state.scanner_running:
                st.session_state.last_scan_time = time.time()
            else:
                if st.session_state.scanner_instance:
                    st.session_state.scanner_instance.stop()
                    st.session_state.scanner_instance = None
    
    with col2:
        if st.button("Start WebSockets" if not st.session_state.websocket_running else "Stop WebSockets"):
            st.session_state.websocket_running = not st.session_state.websocket_running
            if st.session_state.websocket_running:
                # Will be handled in main loop
                pass
            else:
                if st.session_state.scanner_instance:
                    st.session_state.scanner_instance.stop()
    
    pause_label = "Pause Trading" if not st.session_state.trading_paused else "Resume Trading"
    if st.button(pause_label):
        st.session_state.trading_paused = not st.session_state.trading_paused
    
    auto_label = "Enable Auto-Trading" if not st.session_state.auto_trading else "Disable Auto-Trading"
    if st.button(auto_label):
        st.session_state.auto_trading = not st.session_state.auto_trading
        if st.session_state.auto_trading and st.session_state.trading_mode == "Live":
            st.warning("⚠️ Auto-trading enabled with LIVE trading! Trades will execute automatically.")
    
    st.markdown("### Scan Settings")
    st.session_state.scan_interval = st.slider(
        "Scan Interval (seconds)",
        min_value=5,
        max_value=60,
        value=st.session_state.scan_interval
    )
    
    st.markdown("### Algorithm Settings")
    st.session_state.show_advanced = st.checkbox(
        "Show Advanced Settings",
        value=st.session_state.show_advanced
    )
    
    if st.session_state.show_advanced:
        st.session_state.min_profit_threshold = st.slider(
            "Min Profit Threshold (%)",
            min_value=0.1,
            max_value=2.0,
            value=st.session_state.min_profit_threshold,
            step=0.1
        )
        
        st.session_state.safety_margin = st.slider(
            "Safety Margin (%)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.safety_margin,
            step=0.1
        )
        
        st.session_state.max_position_size = st.number_input(
            "Max Position Size (USDT)",
            min_value=100,
            max_value=10000,
            value=st.session_state.max_position_size,
            step=100
        )
        
        st.session_state.parallel_scanning = st.checkbox(
            "Parallel Scanning",
            value=st.session_state.parallel_scanning
        )
        
        if st.session_state.algorithm in ["advanced", "ml"]:
            algorithm_specific = {}
            
            if st.session_state.algorithm == "advanced":
                algorithm_specific["algorithm"] = st.selectbox(
                    "Graph Algorithm",
                    options=["johnson", "bellman-ford", "yen-k-paths"],
                    index=0
                )
                
                algorithm_specific["incremental"] = st.checkbox(
                    "Incremental Updates",
                    value=True
                )
                
                algorithm_specific["subgraph_radius"] = st.slider(
                    "Subgraph Radius",
                    min_value=1,
                    max_value=5,
                    value=2
                )
            
            if st.session_state.algorithm == "ml":
                algorithm_specific["ml_scoring"] = st.checkbox(
                    "ML Loop Scoring",
                    value=True
                )
                
                algorithm_specific["reinforcement_learning"] = st.checkbox(
                    "Reinforcement Learning",
                    value=True
                )
                
                algorithm_specific["depth_threshold"] = st.slider(
                    "Depth Threshold",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    step=0.01
                )
                
                algorithm_specific["volatility_threshold"] = st.slider(
                    "Volatility Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
            
            # Save algorithm-specific settings
            if st.button("Save Algorithm Settings"):
                config = get_default_algorithm_config()
                config.update({
                    'min_profit_threshold': st.session_state.min_profit_threshold,
                    'safety_margin': st.session_state.safety_margin,
                    'max_position_size': st.session_state.max_position_size,
                    'parallel_scanning': st.session_state.parallel_scanning,
                    **algorithm_specific
                })
                save_algorithm_config(config)
                st.success("Settings saved!")
    
    st.markdown("### Display Settings")
    st.session_state.show_profit = st.checkbox("Show Profit %", value=st.session_state.show_profit)
    st.session_state.show_exchange = st.checkbox("Show Exchange", value=st.session_state.show_exchange)
    st.session_state.show_duration = st.checkbox("Show Duration", value=st.session_state.show_duration)
    st.session_state.show_risk = st.checkbox("Show Risk Score", value=st.session_state.show_risk)
    st.session_state.show_risk_adjusted = st.checkbox("Show Risk-Adjusted Profit", value=st.session_state.show_risk_adjusted)
    
    st.markdown("### Timeframe")
    st.session_state.timeframe = st.selectbox(
        "Chart Timeframe",
        options=["30 mins", "2h", "24h"],
        index=["30 mins", "2h", "24h"].index(st.session_state.timeframe)
    )
    
    if st.button("Trigger Manual Scan"):
        st.session_state.trigger_scan = True
    else:
        st.session_state.trigger_scan = False
    
    st.markdown("### Loop Management")
    new_loop = st.text_input("Add Trading Loop (e.g., BTC/USDT,ETH/BTC,ETH/USDT)")
    if st.button("Add Loop") and new_loop:
        loop_symbols = [s.strip() for s in new_loop.split(",")]
        if len(loop_symbols) == 3:
            # Add to loop cache
            if st.session_state.scanner_instance:
                st.session_state.scanner_instance.add_loop_to_cache(loop_symbols)
                st.success(f"Added loop: {loop_symbols}")
            else:
                st.error("Scanner not initialized. Start the scanner first.")
        else:
            st.error("Invalid loop format. Must contain 3 trading pairs.")

# Main content
# Initialize exchange wrappers
def initialize_scanner():
    if not st.session_state.active_exchanges:
        st.warning("No exchanges selected. Please select at least one exchange.")
        return None
    
    exchange_wrappers = {}
    for ex in st.session_state.active_exchanges:
        try:
            if ex == "gateio":
                exchange_wrappers[ex] = GateIOWrapper()
            elif ex == "mexc":
                exchange_wrappers[ex] = MEXCWrapper()
            elif ex == "phemex":
                exchange_wrappers[ex] = PhemexWrapper()
            elif ex == "bitget":
                exchange_wrappers[ex] = BitgetWrapper()
        except Exception as e:
            st.error(f"Error initializing {ex} exchange: {str(e)}")
    
    if not exchange_wrappers:
        st.error("Failed to initialize any exchanges.")
        return None
    
    # Load algorithm config
    config = load_algorithm_config()
    
    # Update with current session state
    config.update({
        'min_profit_threshold': st.session_state.min_profit_threshold,
        'safety_margin': st.session_state.safety_margin,
        'max_position_size': st.session_state.max_position_size,
        'parallel_scanning': st.session_state.parallel_scanning
    })
    
    # Create scanner based on selected algorithm
    if st.session_state.algorithm == "basic":
        from core.scanner import Scanner
        # Basic scanner doesn't have a unified interface, will be handled differently
        return {"wrappers": exchange_wrappers, "type": "basic"}
    elif st.session_state.algorithm == "optimized":
        from core.optimized_scanner import OptimizedArbitrageScanner
        return OptimizedArbitrageScanner(exchange_wrappers, config)
    elif st.session_state.algorithm == "advanced":
        return AdvancedArbitrageScanner(exchange_wrappers, config)
    elif st.session_state.algorithm == "ml":
        return MLArbitrageScanner(exchange_wrappers, config)
    else:
        st.error(f"Unknown algorithm: {st.session_state.algorithm}")
        return None

# Function to run scanner
def run_scanner():
    if not st.session_state.scanner_instance:
        st.session_state.scanner_instance = initialize_scanner()
        if not st.session_state.scanner_instance:
            return []
    
    scanner = st.session_state.scanner_instance
    
    # Start WebSockets if enabled
    if st.session_state.websocket_running:
        if hasattr(scanner, 'start') and callable(scanner.start):
            scanner.start()
    
    # Handle basic scanner differently
    if isinstance(scanner, dict) and scanner.get("type") == "basic":
        from core.scanner import Scanner
        results = []
        for name, wrapper in scanner["wrappers"].items():
            try:
                basic_scanner = Scanner(wrapper)
                # Get loops from cache or use defaults
                loops = []
                try:
                    cache_path = os.path.join("data", "cached_loops.json")
                    if os.path.exists(cache_path):
                        with open(cache_path, 'r') as f:
                            cache = json.load(f)
                            loops = [entry["pairs"] for entry in cache.get("loops", [])]
                except:
                    pass
                
                if not loops:
                    loops = [
                        ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
                        ["ETH/USDT", "XRP/ETH", "XRP/USDT"],
                        ["BTC/USDT", "LTC/BTC", "LTC/USDT"]
                    ]
                
                for loop in loops:
                    try:
                        res = basic_scanner.check_profitability(loop)
                        if res and res.get("profit_percent", 0) > st.session_state.min_profit_threshold:
                            res["exchange"] = name
                            res["loop"] = loop
                            results.append(res)
                    except Exception as e:
                        st.warning(f"Error checking loop {loop} on {name}: {str(e)}")
            except Exception as e:
                st.error(f"Error with scanner for {name}: {str(e)}")
        return results
    else:
        # Use unified interface for other scanners
        return scanner.scan_for_opportunities()

# Function to display scan results
def display_scan_results(results):
    if not results:
        st.info("No profitable opportunities found in this scan.")
        return
    
    st.markdown("<h2 class='sub-header'>🔍 Profitable Opportunities</h2>", unsafe_allow_html=True)
    
    # Sort by profit or ML score
    if st.session_state.algorithm == "ml" and any("ml_score" in r for r in results):
        results = sorted(results, key=lambda x: x.get("ml_score", 0), reverse=True)
    else:
        results = sorted(results, key=lambda x: x.get("adjusted_profit_percent", x.get("profit_percent", 0)), reverse=True)
    
    for i, t in enumerate(results):
        # Format loop display
        loop_str = " → ".join(t['loop'])
        
        # Format profit display based on algorithm
        if "adjusted_profit_percent" in t:
            profit_display = f"{t['adjusted_profit_percent']:.2f}% Profit (Raw: {t['raw_profit_percent']:.2f}%)"
        else:
            profit_display = f"{t['profit_percent']:.2f}% Profit"
        
        # Add ML score if available
        if "ml_score" in t:
            profit_display += f" (ML Score: {t['ml_score']:.2f})"
        
        with st.expander(f"{i+1}. {loop_str} ({profit_display})"):
            cols = st.columns(2)
            
            with cols[0]:
                if st.session_state.show_exchange:
                    st.write(f"**Exchange:** {t['exchange'].upper()}")
                    
                if st.session_state.show_profit:
                    if "adjusted_profit_percent" in t:
                        st.write(f"**Raw Profit:** {t['raw_profit_percent']:.2f}%")
                        st.write(f"**Adjusted Profit:** {t['adjusted_profit_percent']:.2f}%")
                    else:
                        st.write(f"**Profit:** {t['profit_percent']:.2f}%")
                        
                if st.session_state.show_duration:
                    duration = t.get('execution_time_ms', t.get('duration', 0))
                    st.write(f"**Duration:** {duration} ms")
                    
                # Show risk metrics for advanced algorithms
                if st.session_state.algorithm in ["optimized", "advanced", "ml"]:
                    if st.session_state.show_risk and "risk_score" in t:
                        st.write(f"**Risk Score:** {t['risk_score']:.4f}")
                        if "risk_reward_ratio" in t:
                            st.write(f"**Risk-Reward Ratio:** {t['risk_reward_ratio']:.2f}")
                        
                    if st.session_state.show_risk_adjusted and "risk_adjusted_profit" in t:
                        st.write(f"**Risk-Adjusted Profit:** {t['risk_adjusted_profit']:.2f}%")
                        
                    # Show position size
                    if "position_size" in t:
                        st.write(f"**Optimal Position Size:** {t['position_size']:.2f} USDT")
            
            with cols[1]:
                # Execute button
                if st.button(f"Execute Trade #{i+1}", key=f"exec_{i}"):
                    if st.session_state.trading_paused:
                        st.error("Trading is paused. Resume trading to execute.")
                    else:
                        if st.session_state.trading_mode == "Live":
                            st.warning("Executing LIVE trade...")
                            # Execute real trade
                            if hasattr(st.session_state.scanner_instance, 'execute_arbitrage'):
                                result = st.session_state.scanner_instance.execute_arbitrage(t)
                                if result['success']:
                                    st.success(f"Trade executed successfully! Profit: {result['profit_percent']:.2f}%")
                                else:
                                    st.error(f"Trade execution failed: {result.get('error', 'Unknown error')}")
                            else:
                                st.error("This algorithm doesn't support trade execution.")
                        else:
                            st.success("Paper trade executed successfully!")
                            # Record paper trade in history
                            trade_record = {
                                'timestamp': time.time(),
                                'exchange': t['exchange'],
                                'loop': t['loop'],
                                'position_size': t.get('position_size', 100),
                                'expected_profit_percent': t.get('adjusted_profit_percent', t.get('profit_percent', 0)),
                                'actual_profit_percent': t.get('adjusted_profit_percent', t.get('profit_percent', 0)),
                                'execution_time_ms': t.get('execution_time_ms', 0),
                                'success': True,
                                'mode': 'Paper'
                            }
                            
                            # Add to trade history
                            st.session_state.trade_history.append(trade_record)
                            
                            # Save to file
                            try:
                                history_path = os.path.join("data", "trade_history.json")
                                os.makedirs(os.path.dirname(history_path), exist_ok=True)
                                
                                history = []
                                if os.path.exists(history_path):
                                    with open(history_path, 'r') as f:
                                        history = json.load(f)
                                
                                history.append(trade_record)
                                
                                with open(history_path, 'w') as f:
                                    json.dump(history, f, indent=2)
                            except Exception as e:
                                st.error(f"Error saving trade history: {str(e)}")
                
                # Show trade simulation details in a collapsible section
                if "trade_simulation" in t:
                    with st.expander("Trade Simulation Details"):
                        sim = t['trade_simulation']
                        st.write("**First Trade:**")
                        st.json(sim['first_trade'])
                        st.write("**Second Trade:**")
                        st.json(sim['second_trade'])
                        st.write("**Final Trade:**")
                        st.json(sim['final_trade'])

# Function to display performance metrics
def display_performance_metrics():
    st.markdown("<h2 class='sub-header'>📊 Performance Metrics</h2>", unsafe_allow_html=True)
    
    # Get performance metrics
    metrics = {}
    if hasattr(st.session_state.scanner_instance, 'get_performance_report'):
        try:
            report = st.session_state.scanner_instance.get_performance_report()
            metrics = report.get('metrics', {})
            recommendations = report.get('recommendations', [])
            circuit_breakers = report.get('circuit_breakers', {})
        except:
            pass
    
    if not metrics:
        # Calculate basic metrics from trade history
        if st.session_state.trade_history:
            successful_trades = [t for t in st.session_state.trade_history if t.get('success', False)]
            total_profit = sum(t.get('actual_profit_percent', 0) * t.get('position_size', 100) / 100 
                              for t in successful_trades)
            win_rate = len(successful_trades) / len(st.session_state.trade_history) if st.session_state.trade_history else 0
            avg_profit = sum(t.get('actual_profit_percent', 0) for t in successful_trades) / len(successful_trades) if successful_trades else 0
            
            metrics = {
                'total_profit': total_profit,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'trade_count': len(st.session_state.trade_history)
            }
    
    if metrics:
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Profit", f"{metrics.get('total_profit', 0):.2f} USDT")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Win Rate", f"{metrics.get('win_rate', 0) * 100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg Profit per Trade", f"{metrics.get('avg_profit', 0):.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Trades", f"{metrics.get('trade_count', 0)}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display recommendations if available
        if 'recommendations' in locals() and recommendations:
            with st.expander("System Recommendations"):
                for rec in recommendations:
                    st.info(rec)
        
        # Display circuit breaker status if available
        if 'circuit_breakers' in locals() and circuit_breakers:
            with st.expander("Circuit Breaker Status"):
                for exchange, status in circuit_breakers.items():
                    if status:
                        st.error(f"{exchange.upper()}: OPEN - Connection issues detected")
                    else:
                        st.success(f"{exchange.upper()}: CLOSED - Operating normally")
    else:
        st.info("No performance metrics available yet. Execute some trades to see metrics.")

# Function to display trade history
def display_trade_history():
    st.markdown("<h2 class='sub-header'>📜 Trade History</h2>", unsafe_allow_html=True)
    
    # Load trade history from file if not in session state
    if not st.session_state.trade_history:
        try:
            history_path = os.path.join("data", "trade_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    st.session_state.trade_history = json.load(f)
        except Exception as e:
            st.error(f"Error loading trade history: {str(e)}")
    
    if not st.session_state.trade_history:
        st.info("No trade history available yet.")
        return
    
    # Convert to DataFrame for display
    try:
        df = pd.DataFrame(st.session_state.trade_history)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Format for display
        display_df = df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['loop'] = display_df['loop'].apply(lambda x: " → ".join(x))
        display_df['profit'] = display_df['actual_profit_percent'].apply(lambda x: f"{x:.2f}%")
        
        # Select columns for display
        display_cols = ['timestamp', 'exchange', 'loop', 'profit', 'success', 'mode']
        if 'mode' not in display_df.columns:
            display_df['mode'] = 'Paper'
            
        st.dataframe(display_df[display_cols])
        
        # Display charts
        st.markdown("<h3 class='sub-header'>Trade Performance Charts</h3>", unsafe_allow_html=True)
        
        # Filter by timeframe
        cutoff = {
            "30 mins": datetime.now() - timedelta(minutes=30),
            "2h": datetime.now() - timedelta(hours=2),
            "24h": datetime.now() - timedelta(hours=24)
        }[st.session_state.timeframe]
        
        filtered_df = df[df['timestamp'] > cutoff]
        
        if filtered_df.empty:
            st.info(f"No data available for the selected timeframe ({st.session_state.timeframe}).")
            return
        
        # Create tabs for different charts
        tab1, tab2, tab3 = st.tabs(["Profit Over Time", "Exchange Distribution", "Success Rate"])
        
        with tab1:
            # Profit over time chart
            fig = px.line(
                filtered_df, 
                x='timestamp', 
                y='actual_profit_percent',
                title='Profit Percentage Over Time',
                labels={'actual_profit_percent': 'Profit %', 'timestamp': 'Time'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Exchange distribution chart
            exchange_counts = filtered_df['exchange'].value_counts().reset_index()
            exchange_counts.columns = ['Exchange', 'Count']
            
            fig = px.bar(
                exchange_counts,
                x='Exchange',
                y='Count',
                title='Trades by Exchange',
                color='Exchange'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Success rate chart
            success_counts = filtered_df['success'].value_counts().reset_index()
            success_counts.columns = ['Success', 'Count']
            
            fig = px.pie(
                success_counts,
                values='Count',
                names='Success',
                title='Trade Success Rate',
                color='Success',
                color_discrete_map={True: '#4CAF50', False: '#F44336'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying trade history: {str(e)}")

# Function to display system status
def display_system_status():
    st.markdown("<h2 class='sub-header'>🖥️ System Status</h2>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    with cols[0]:
        scanner_status = "🟢 Running" if st.session_state.scanner_running else "🔴 Stopped"
        st.markdown(f"**Scanner:** {scanner_status}")
    
    with cols[1]:
        websocket_status = "🟢 Connected" if st.session_state.websocket_running else "🔴 Disconnected"
        st.markdown(f"**WebSockets:** {websocket_status}")
    
    with cols[2]:
        trading_status = "🔴 Paused" if st.session_state.trading_paused else "🟢 Active"
        st.markdown(f"**Trading:** {trading_status}")
    
    with cols[3]:
        auto_status = "🟢 Enabled" if st.session_state.auto_trading else "🔴 Disabled"
        st.markdown(f"**Auto-Trading:** {auto_status}")
    
    # Last scan time
    if st.session_state.last_scan_time:
        last_scan = datetime.fromtimestamp(st.session_state.last_scan_time).strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f"**Last Scan:** {last_scan}")
    
    # Active exchanges
    st.markdown(f"**Active Exchanges:** {', '.join([ex.upper() for ex in st.session_state.active_exchanges])}")
    
    # Algorithm
    st.markdown(f"**Algorithm:** {algorithm_options[st.session_state.algorithm]}")
    
    # Trading mode
    mode_color = "🔴" if st.session_state.trading_mode == "Live" else "🟢"
    st.markdown(f"**Trading Mode:** {mode_color} {st.session_state.trading_mode}")

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    # Status and controls
    display_system_status()
    
    # Scan results
    if st.session_state.scanner_running or st.session_state.trigger_scan:
        with st.spinner("Scanning for arbitrage opportunities..."):
            results = run_scanner()
            st.session_state.scan_results = results
            st.session_state.last_scan_time = time.time()
    
    display_scan_results(st.session_state.scan_results)

with col2:
    # Performance metrics
    display_performance_metrics()

# Trade history (full width)
display_trade_history()

# Auto-trading and periodic scanning
if st.session_state.scanner_running:
    # Check if it's time for a new scan
    current_time = time.time()
    if st.session_state.last_scan_time and (current_time - st.session_state.last_scan_time) >= st.session_state.scan_interval:
        st.session_state.last_scan_time = current_time
        st.experimental_rerun()
    
    # Auto-trading logic
    if st.session_state.auto_trading and not st.session_state.trading_paused and st.session_state.scan_results:
        # Get the best opportunity
        best_opportunity = None
        
        if st.session_state.algorithm == "ml" and any("ml_score" in r for r in st.session_state.scan_results):
            # Sort by ML score
            sorted_results = sorted(st.session_state.scan_results, key=lambda x: x.get("ml_score", 0), reverse=True)
            if sorted_results:
                best_opportunity = sorted_results[0]
        else:
            # Sort by adjusted profit
            sorted_results = sorted(
                st.session_state.scan_results, 
                key=lambda x: x.get("adjusted_profit_percent", x.get("profit_percent", 0)), 
                reverse=True
            )
            if sorted_results:
                best_opportunity = sorted_results[0]
        
        if best_opportunity:
            # Check if profit exceeds threshold
            profit = best_opportunity.get("adjusted_profit_percent", best_opportunity.get("profit_percent", 0))
            
            if profit > st.session_state.min_profit_threshold:
                # Execute trade
                if st.session_state.trading_mode == "Live":
                    # Execute real trade
                    if hasattr(st.session_state.scanner_instance, 'execute_arbitrage'):
                        result = st.session_state.scanner_instance.execute_arbitrage(best_opportunity)
                        # Result will be processed on next run
                else:
                    # Paper trade
                    trade_record = {
                        'timestamp': time.time(),
                        'exchange': best_opportunity['exchange'],
                        'loop': best_opportunity['loop'],
                        'position_size': best_opportunity.get('position_size', 100),
                        'expected_profit_percent': profit,
                        'actual_profit_percent': profit,
                        'execution_time_ms': best_opportunity.get('execution_time_ms', 0),
                        'success': True,
                        'mode': 'Paper (Auto)'
                    }
                    
                    # Add to trade history
                    st.session_state.trade_history.append(trade_record)
                    
                    # Save to file
                    try:
                        history_path = os.path.join("data", "trade_history.json")
                        os.makedirs(os.path.dirname(history_path), exist_ok=True)
                        
                        history = []
                        if os.path.exists(history_path):
                            with open(history_path, 'r') as f:
                                history = json.load(f)
                        
                        history.append(trade_record)
                        
                        with open(history_path, 'w') as f:
                            json.dump(history, f, indent=2)
                    except Exception as e:
                        pass  # Silently fail, will log on next run
