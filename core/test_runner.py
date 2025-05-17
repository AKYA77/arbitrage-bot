import os
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_runner")

class TestRunner:
    """
    Test runner for validating arbitrage bot functionality.
    """
    
    def __init__(self):
        """Initialize the test runner."""
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "test_cases": []
        }
        
        # Ensure logs directory exists
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Mock Streamlit session state for dashboard tests
        self._setup_streamlit_mock()
    
    def _setup_streamlit_mock(self):
        """Set up mock for Streamlit to avoid session state errors."""
        try:
            import sys
            import types
            
            # Create mock module if streamlit is not available
            if 'streamlit' not in sys.modules:
                mock_streamlit = types.ModuleType('streamlit')
                mock_session_state = types.SimpleNamespace()
                
                # Add required attributes to session state
                mock_session_state.algorithm = "optimized"
                mock_session_state.exchanges = ["gateio", "mexc"]
                mock_session_state.trading_mode = "paper"
                mock_session_state.auto_trading = False
                mock_session_state.min_profit = 0.5
                mock_session_state.safety_margin = 0.2
                mock_session_state.max_position = 1000
                mock_session_state.show_advanced = False
                
                # Add session_state to streamlit mock
                mock_streamlit.session_state = mock_session_state
                
                # Add required streamlit functions
                def mock_sidebar(*args, **kwargs):
                    return types.SimpleNamespace()
                
                def mock_empty(*args, **kwargs):
                    return types.SimpleNamespace()
                
                def mock_container(*args, **kwargs):
                    return types.SimpleNamespace()
                
                def mock_write(*args, **kwargs):
                    pass
                
                def mock_markdown(*args, **kwargs):
                    pass
                
                def mock_selectbox(*args, **kwargs):
                    return args[2][0] if len(args) > 2 and isinstance(args[2], list) and len(args[2]) > 0 else None
                
                def mock_multiselect(*args, **kwargs):
                    return args[2] if len(args) > 2 and isinstance(args[2], list) else []
                
                def mock_number_input(*args, **kwargs):
                    return kwargs.get('value', 0)
                
                def mock_checkbox(*args, **kwargs):
                    return kwargs.get('value', False)
                
                def mock_button(*args, **kwargs):
                    return False
                
                def mock_error(*args, **kwargs):
                    pass
                
                def mock_success(*args, **kwargs):
                    pass
                
                def mock_info(*args, **kwargs):
                    pass
                
                def mock_metric(*args, **kwargs):
                    pass
                
                # Add all mock functions to streamlit mock
                mock_streamlit.sidebar = mock_sidebar
                mock_streamlit.empty = mock_empty
                mock_streamlit.container = mock_container
                mock_streamlit.write = mock_write
                mock_streamlit.markdown = mock_markdown
                mock_streamlit.selectbox = mock_selectbox
                mock_streamlit.multiselect = mock_multiselect
                mock_streamlit.number_input = mock_number_input
                mock_streamlit.checkbox = mock_checkbox
                mock_streamlit.button = mock_button
                mock_streamlit.error = mock_error
                mock_streamlit.success = mock_success
                mock_streamlit.info = mock_info
                mock_streamlit.metric = mock_metric
                
                # Register the mock module
                sys.modules['streamlit'] = mock_streamlit
                
                logger.info("Streamlit mock set up successfully")
        except Exception as e:
            logger.warning(f"Failed to set up Streamlit mock: {str(e)}")
    
    def run_all_tests(self) -> Dict:
        """
        Run all test cases.
        
        Returns:
            Dict with test results
        """
        logger.info("Starting test suite")
        
        # Run module tests
        self._test_config_module()
        self._test_loop_manager()
        self._test_graph_algorithms()
        self._test_exchange_wrappers()
        self._test_scanners()
        self._test_dashboard()
        self._test_integration()
        
        # Log summary
        logger.info(f"Test suite completed: {self.results['passed']} passed, {self.results['failed']} failed, {self.results['skipped']} skipped")
        
        return self.results
    
    def _record_result(self, test_name: str, passed: bool, error: Optional[str] = None, skipped: bool = False) -> None:
        """
        Record a test result.
        
        Args:
            test_name: Name of the test
            passed: Whether the test passed
            error: Optional error message
            skipped: Whether the test was skipped
        """
        self.results["total_tests"] += 1
        
        if skipped:
            self.results["skipped"] += 1
            logger.info(f"SKIPPED: {test_name}")
        elif passed:
            self.results["passed"] += 1
            logger.info(f"PASSED: {test_name}")
        else:
            self.results["failed"] += 1
            logger.error(f"FAILED: {test_name} - {error}")
        
        self.results["test_cases"].append({
            "name": test_name,
            "passed": passed,
            "skipped": skipped,
            "error": error,
            "timestamp": None  # Will be filled in when saving results
        })
    
    def _test_config_module(self) -> None:
        """Test the configuration module."""
        logger.info("Testing configuration module")
        
        try:
            from core.config import load_api_keys, get_default_algorithm_config, save_algorithm_config, load_algorithm_config
            
            # Test loading API keys
            api_keys = load_api_keys()
            self._record_result("Config - Load API Keys", isinstance(api_keys, dict))
            
            # Test default algorithm config
            default_config = get_default_algorithm_config()
            self._record_result("Config - Default Algorithm Config", 
                               isinstance(default_config, dict) and "min_profit_threshold" in default_config)
            
            # Test saving and loading algorithm config
            test_config = {"test_key": "test_value", "min_profit_threshold": 0.5}
            save_algorithm_config(test_config)
            loaded_config = load_algorithm_config()
            self._record_result("Config - Save/Load Algorithm Config", 
                               isinstance(loaded_config, dict) and "test_key" in loaded_config)
            
        except Exception as e:
            self._record_result("Config Module Tests", False, str(e))
    
    def _test_loop_manager(self) -> None:
        """Test the loop manager module."""
        logger.info("Testing loop manager module")
        
        try:
            from core.loop_manager import LoopManager
            
            # Initialize loop manager
            loop_manager = LoopManager()
            
            # Test adding a loop
            test_loop = ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
            added = loop_manager.add_loop(test_loop, {"test": True})
            self._record_result("Loop Manager - Add Loop", added)
            
            # Test getting loops
            loops = loop_manager.get_loops()
            self._record_result("Loop Manager - Get Loops", 
                               isinstance(loops, list) and len(loops) > 0)
            
            # Test getting loop pairs
            loop_pairs = loop_manager.get_loop_pairs()
            self._record_result("Loop Manager - Get Loop Pairs", 
                               isinstance(loop_pairs, list) and len(loop_pairs) > 0)
            
            # Test updating loop metrics
            loop_id = "-".join(test_loop)
            loop_manager.update_loop_metrics(loop_id, {"success": True, "profit": 1.5})
            metrics = loop_manager.get_loop_metrics(loop_id)
            self._record_result("Loop Manager - Update/Get Metrics", 
                               isinstance(metrics, dict) and metrics["successes"] == 1)
            
            # Test removing a loop
            removed = loop_manager.remove_loop(loop_id)
            self._record_result("Loop Manager - Remove Loop", removed)
            
        except Exception as e:
            self._record_result("Loop Manager Tests", False, str(e))
    
    def _test_graph_algorithms(self) -> None:
        """Test the graph algorithms module."""
        logger.info("Testing graph algorithms module")
        
        try:
            from core.graph_algorithms import ArbitrageGraph, JohnsonAlgorithm, BellmanFordAlgorithm, YenKShortestPaths
            
            # Initialize graph
            graph = ArbitrageGraph()
            
            # Add edges
            graph.add_edge("USDT", "BTC", "test_exchange", 1/40000, 1000)
            graph.add_edge("BTC", "ETH", "test_exchange", 15, 10)
            graph.add_edge("ETH", "USDT", "test_exchange", 3000, 100)
            
            # Test Johnson's algorithm
            johnson_cycles = JohnsonAlgorithm.find_negative_cycles(graph)
            self._record_result("Graph Algorithms - Johnson's Algorithm", 
                               isinstance(johnson_cycles, list))
            
            # Test Bellman-Ford algorithm
            bellman_ford_cycles = BellmanFordAlgorithm.find_negative_cycles(graph)
            self._record_result("Graph Algorithms - Bellman-Ford Algorithm", 
                               isinstance(bellman_ford_cycles, list))
            
            # Test Yen's K-shortest paths algorithm
            yen_paths = YenKShortestPaths.find_k_shortest_paths(graph, "USDT", "BTC", 3)
            self._record_result("Graph Algorithms - Yen's K-Shortest Paths", 
                               isinstance(yen_paths, list))
            
        except Exception as e:
            self._record_result("Graph Algorithms Tests", False, str(e))
    
    def _test_exchange_wrappers(self) -> None:
        """Test the exchange wrapper modules."""
        logger.info("Testing exchange wrapper modules")
        
        # Test GateIO wrapper
        try:
            from wrappers.gateio_wrapper import GateIOWrapper
            
            wrapper = GateIOWrapper()
            
            # Test getting order book
            order_book = wrapper.get_order_book("BTC/USDT", 5)
            valid_order_book = (isinstance(order_book, dict) and 
                               "bids" in order_book and "asks" in order_book)
            
            self._record_result("GateIO Wrapper - Get Order Book", valid_order_book)
            
            # Test getting fee rate
            fee_rate = wrapper.get_fee_rate()
            self._record_result("GateIO Wrapper - Get Fee Rate", 
                               isinstance(fee_rate, float) and 0 < fee_rate < 0.1)
            
            # Test WebSocket URL
            ws_url = wrapper.get_websocket_url()
            self._record_result("GateIO Wrapper - WebSocket URL", 
                               isinstance(ws_url, str) and ws_url.startswith("wss://"))
            
            # Test WebSocket subscribe message
            subscribe_msg = wrapper.get_websocket_subscribe_message()
            self._record_result("GateIO Wrapper - WebSocket Subscribe Message", 
                               isinstance(subscribe_msg, str) and len(subscribe_msg) > 0)
            
        except Exception as e:
            self._record_result("GateIO Wrapper Tests", False, str(e))
        
        # Test MEXC wrapper
        try:
            from wrappers.mexc_wrapper import MEXCWrapper
            
            wrapper = MEXCWrapper()
            
            # Test getting order book
            order_book = wrapper.get_order_book("BTC/USDT", 5)
            valid_order_book = (isinstance(order_book, dict) and 
                               "bids" in order_book and "asks" in order_book)
            
            self._record_result("MEXC Wrapper - Get Order Book", valid_order_book)
            
            # Test getting fee rate
            fee_rate = wrapper.get_fee_rate()
            self._record_result("MEXC Wrapper - Get Fee Rate", 
                               isinstance(fee_rate, float) and 0 < fee_rate < 0.1)
            
            # Test WebSocket URL
            ws_url = wrapper.get_websocket_url()
            self._record_result("MEXC Wrapper - WebSocket URL", 
                               isinstance(ws_url, str) and ws_url.startswith("wss://"))
            
            # Test WebSocket subscribe message
            subscribe_msg = wrapper.get_websocket_subscribe_message()
            self._record_result("MEXC Wrapper - WebSocket Subscribe Message", 
                               isinstance(subscribe_msg, str) and len(subscribe_msg) > 0)
            
        except Exception as e:
            self._record_result("MEXC Wrapper Tests", False, str(e))
        
        # Test other wrappers (similar pattern)
        for wrapper_name in ["Phemex", "Bitget"]:
            try:
                if wrapper_name == "Phemex":
                    from wrappers.phemex_wrapper import PhemexWrapper as Wrapper
                else:  # Bitget
                    from wrappers.bitget_wrapper import BitgetWrapper as Wrapper
                
                wrapper = Wrapper()
                
                # Test getting order book
                order_book = wrapper.get_order_book("BTC/USDT", 5)
                valid_order_book = (isinstance(order_book, dict) and 
                                   "bids" in order_book and "asks" in order_book)
                
                self._record_result(f"{wrapper_name} Wrapper - Get Order Book", valid_order_book)
                
                # Test getting fee rate
                fee_rate = wrapper.get_fee_rate()
                self._record_result(f"{wrapper_name} Wrapper - Get Fee Rate", 
                                   isinstance(fee_rate, float) and 0 < fee_rate < 0.1)
                
                # Test WebSocket URL
                ws_url = wrapper.get_websocket_url()
                self._record_result(f"{wrapper_name} Wrapper - WebSocket URL", 
                                   isinstance(ws_url, str) and ws_url.startswith("wss://"))
                
                # Test WebSocket subscribe message
                subscribe_msg = wrapper.get_websocket_subscribe_message()
                self._record_result(f"{wrapper_name} Wrapper - WebSocket Subscribe Message", 
                                   isinstance(subscribe_msg, str) and len(subscribe_msg) > 0)
                
            except Exception as e:
                self._record_result(f"{wrapper_name} Wrapper Tests", False, str(e))
    
    def _test_scanners(self) -> None:
        """Test the scanner modules."""
        logger.info("Testing scanner modules")
        
        # Test basic scanner
        try:
            from core.scanner import Scanner
            from wrappers.gateio_wrapper import GateIOWrapper
            
            wrapper = GateIOWrapper()
            scanner = Scanner(wrapper)
            
            # Test checking profitability
            loop = ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
            result = scanner.check_profitability(loop)
            
            self._record_result("Basic Scanner - Check Profitability", 
                               isinstance(result, dict) or result is None)
            
        except Exception as e:
            self._record_result("Basic Scanner Tests", False, str(e))
        
        # Test optimized scanner
        try:
            from core.optimized_scanner import OptimizedArbitrageScanner
            
            # Create exchange wrappers
            from wrappers.gateio_wrapper import GateIOWrapper
            exchange_wrappers = {"gateio": GateIOWrapper()}
            
            # Initialize scanner
            scanner = OptimizedArbitrageScanner(exchange_wrappers)
            
            # Test scanning for opportunities
            opportunities = scanner.scan_for_opportunities()
            
            self._record_result("Optimized Scanner - Scan For Opportunities", 
                               isinstance(opportunities, list))
            
        except Exception as e:
            self._record_result("Optimized Scanner Tests", False, str(e))
        
        # Test advanced scanner
        try:
            from core.advanced_scanner import AdvancedArbitrageScanner
            
            # Create exchange wrappers
            from wrappers.gateio_wrapper import GateIOWrapper
            exchange_wrappers = {"gateio": GateIOWrapper()}
            
            # Initialize scanner with config that includes parallel_scanning
            config = {
                'min_profit_threshold': 0.5,
                'safety_margin': 0.2,
                'max_slippage': 0.3,
                'order_book_depth': 15,
                'max_position_size': 1000,
                'min_position_size': 10,
                'parallel_scanning': True,
                'algorithm': 'johnson',
                'incremental': True,
                'subgraph_radius': 2,
                'depth_threshold': 0.1,
                'volatility_threshold': 0.5
            }
            
            # Initialize scanner
            scanner = AdvancedArbitrageScanner(exchange_wrappers, config)
            
            # Test scanning for opportunities
            opportunities = scanner.scan_for_opportunities()
            
            self._record_result("Advanced Scanner - Scan For Opportunities", 
                               isinstance(opportunities, list))
            
            # Test loop cache
            self._record_result("Advanced Scanner - Loop Cache", 
                               isinstance(scanner.loop_cache, dict) and "loops" in scanner.loop_cache)
            
        except Exception as e:
            self._record_result("Advanced Scanner Tests", False, str(e))
        
        # Test ML scanner (skip actual execution due to complexity)
        try:
            from core.ml_scanner import MLArbitrageScanner
            
            # Just test import
            self._record_result("ML Scanner - Import", True)
            
            # Skip actual execution
            self._record_result("ML Scanner - Execution", True, skipped=True)
            
        except Exception as e:
            self._record_result("ML Scanner Tests", False, str(e))
    
    def _test_dashboard(self) -> None:
        """Test the dashboard module."""
        logger.info("Testing dashboard module")
        
        try:
            # Test dashboard import
            import dashboard
            
            # Just test import
            self._record_result("Dashboard - Import", True)
            
            # Skip actual execution
            self._record_result("Dashboard - Execution", True, skipped=True)
            
        except Exception as e:
            self._record_result("Dashboard Tests", False, str(e))
    
    def _test_integration(self) -> None:
        """Test integration between modules."""
        logger.info("Testing integration between modules")
        
        try:
            # Test config with scanner integration
            from core.config import load_algorithm_config
            from core.advanced_scanner import AdvancedArbitrageScanner
            from wrappers.gateio_wrapper import GateIOWrapper
            
            # Ensure config has parallel_scanning
            config = load_algorithm_config()
            if 'parallel_scanning' not in config:
                config['parallel_scanning'] = True
            
            exchange_wrappers = {"gateio": GateIOWrapper()}
            scanner = AdvancedArbitrageScanner(exchange_wrappers, config)
            
            self._record_result("Integration - Config with Scanner", 
                               isinstance(scanner.config, dict) and "min_profit_threshold" in scanner.config)
            
            # Test loop manager with scanner integration
            from core.loop_manager import LoopManager
            
            loop_manager = LoopManager()
            test_loop = ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
            loop_manager.add_loop(test_loop)
            
            loops = loop_manager.get_loop_pairs()
            opportunities = scanner.scan_for_opportunities(loops)
            
            self._record_result("Integration - Loop Manager with Scanner", 
                               isinstance(opportunities, list))
            
            # Test WebSocket functionality (skip actual connection)
            self._record_result("Integration - WebSocket Functionality", True, skipped=True)
            
        except Exception as e:
            self._record_result("Integration Tests", False, str(e))
        
        # Test end-to-end flow (skip actual execution)
        self._record_result("Integration - End-to-End Flow", True, skipped=True)

if __name__ == "__main__":
    # Run tests
    runner = TestRunner()
    results = runner.run_all_tests()
    
    # Save results to file
    import time
    
    # Add timestamps to results
    for test_case in results["test_cases"]:
        test_case["timestamp"] = time.time()
    
    results_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Tests completed: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped")
    print(f"Results saved to {results_file}")
