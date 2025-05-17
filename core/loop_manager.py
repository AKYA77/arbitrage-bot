import os
import json
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("loop_manager")

class LoopManager:
    """
    Manages trading loops for the arbitrage bot.
    Handles caching, discovery, and prioritization of loops.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the loop manager.
        
        Args:
            data_dir: Optional directory for storing loop data
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.cache_file = os.path.join(self.data_dir, "cached_loops.json")
        self.loop_cache = self._load_cache()
        
        # Performance metrics for loops
        self.loop_metrics = {}
        
        # Default loops if none are cached
        self.default_loops = [
            ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
            ["ETH/USDT", "XRP/ETH", "XRP/USDT"],
            ["BTC/USDT", "LTC/BTC", "LTC/USDT"],
            ["BTC/USDT", "XRP/BTC", "XRP/USDT"],
            ["ETH/USDT", "LTC/ETH", "LTC/USDT"]
        ]
    
    def _load_cache(self) -> Dict:
        """
        Load cached loops from disk.
        
        Returns:
            Dict containing cached loops and metadata
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
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
    
    def save_cache(self) -> None:
        """Save the loop cache to disk."""
        try:
            # Update timestamp
            self.loop_cache["last_updated"] = time.time()
            
            # Increment version
            self.loop_cache["version"] = self.loop_cache.get("version", 0) + 1
            
            # Write to temporary file first
            temp_file = self.cache_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.loop_cache, f, indent=2)
            
            # Rename to actual file (atomic operation)
            os.replace(temp_file, self.cache_file)
            
            logger.debug("Loop cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving loop cache: {str(e)}")
    
    def add_loop(self, loop: List[str], metadata: Dict = None) -> bool:
        """
        Add a new loop to the cache if it doesn't exist.
        
        Args:
            loop: List of trading pairs forming a loop
            metadata: Additional metadata about the loop
            
        Returns:
            True if the loop was added, False if it already existed
        """
        # Generate a unique ID for the loop
        loop_id = "-".join(loop)
        
        # Check if loop already exists
        existing_ids = {l.get("id") for l in self.loop_cache["loops"]}
        if loop_id in existing_ids:
            return False
        
        # Create loop entry with timestamp
        loop_entry = {
            "id": loop_id,
            "pairs": loop,
            "discovered_at": time.time(),
            "metadata": metadata or {}
        }
        
        # Add to cache
        self.loop_cache["loops"].append(loop_entry)
        
        # Save cache
        self.save_cache()
        
        return True
    
    def remove_loop(self, loop_id: str) -> bool:
        """
        Remove a loop from the cache.
        
        Args:
            loop_id: ID of the loop to remove
            
        Returns:
            True if the loop was removed, False if it wasn't found
        """
        original_length = len(self.loop_cache["loops"])
        self.loop_cache["loops"] = [l for l in self.loop_cache["loops"] if l.get("id") != loop_id]
        
        if len(self.loop_cache["loops"]) < original_length:
            self.save_cache()
            return True
        
        return False
    
    def get_loops(self, limit: Optional[int] = None, sort_by: str = "discovered_at") -> List[Dict]:
        """
        Get loops from the cache.
        
        Args:
            limit: Optional maximum number of loops to return
            sort_by: Field to sort by (discovered_at, profit, etc.)
            
        Returns:
            List of loop entries
        """
        loops = self.loop_cache["loops"]
        
        # Sort loops
        if sort_by == "discovered_at":
            # Sort by discovery time (newest first)
            sorted_loops = sorted(loops, key=lambda x: x.get("discovered_at", 0), reverse=True)
        elif sort_by == "profit":
            # Sort by profit (highest first)
            sorted_loops = sorted(
                loops, 
                key=lambda x: x.get("metadata", {}).get("profit_percent", 0), 
                reverse=True
            )
        elif sort_by == "score":
            # Sort by ML score if available (highest first)
            sorted_loops = sorted(
                loops, 
                key=lambda x: x.get("metadata", {}).get("score", 0), 
                reverse=True
            )
        else:
            # Default to discovery time
            sorted_loops = sorted(loops, key=lambda x: x.get("discovered_at", 0), reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            return sorted_loops[:limit]
        
        return sorted_loops
    
    def get_loop_pairs(self, limit: Optional[int] = None, sort_by: str = "discovered_at") -> List[List[str]]:
        """
        Get loop trading pairs from the cache.
        
        Args:
            limit: Optional maximum number of loops to return
            sort_by: Field to sort by (discovered_at, profit, etc.)
            
        Returns:
            List of loop trading pairs
        """
        loops = self.get_loops(limit, sort_by)
        
        # Extract just the pairs
        loop_pairs = [l["pairs"] for l in loops]
        
        # If no loops in cache, use defaults
        if not loop_pairs:
            return self.default_loops[:limit] if limit else self.default_loops
        
        return loop_pairs
    
    def update_loop_metrics(self, loop_id: str, metrics: Dict) -> None:
        """
        Update performance metrics for a loop.
        
        Args:
            loop_id: ID of the loop
            metrics: Performance metrics to update
        """
        if loop_id not in self.loop_metrics:
            self.loop_metrics[loop_id] = {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "total_profit": 0,
                "avg_profit": 0,
                "last_execution": 0,
                "last_profit": 0
            }
        
        # Update metrics
        self.loop_metrics[loop_id]["executions"] += 1
        self.loop_metrics[loop_id]["last_execution"] = time.time()
        
        if metrics.get("success", False):
            self.loop_metrics[loop_id]["successes"] += 1
            profit = metrics.get("profit", 0)
            self.loop_metrics[loop_id]["total_profit"] += profit
            self.loop_metrics[loop_id]["last_profit"] = profit
            self.loop_metrics[loop_id]["avg_profit"] = (
                self.loop_metrics[loop_id]["total_profit"] / 
                self.loop_metrics[loop_id]["successes"]
            )
        else:
            self.loop_metrics[loop_id]["failures"] += 1
        
        # Update loop metadata
        for loop in self.loop_cache["loops"]:
            if loop.get("id") == loop_id:
                if "metadata" not in loop:
                    loop["metadata"] = {}
                
                loop["metadata"]["metrics"] = self.loop_metrics[loop_id]
                break
        
        # Save cache
        self.save_cache()
    
    def get_loop_by_id(self, loop_id: str) -> Optional[Dict]:
        """
        Get a loop by its ID.
        
        Args:
            loop_id: ID of the loop
            
        Returns:
            Loop entry or None if not found
        """
        for loop in self.loop_cache["loops"]:
            if loop.get("id") == loop_id:
                return loop
        
        return None
    
    def get_loop_metrics(self, loop_id: str) -> Optional[Dict]:
        """
        Get performance metrics for a loop.
        
        Args:
            loop_id: ID of the loop
            
        Returns:
            Metrics or None if not found
        """
        return self.loop_metrics.get(loop_id)
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """
        Get performance metrics for all loops.
        
        Returns:
            Dict mapping loop IDs to metrics
        """
        return self.loop_metrics
    
    def get_best_performing_loops(self, limit: int = 5) -> List[Dict]:
        """
        Get the best performing loops based on average profit.
        
        Args:
            limit: Maximum number of loops to return
            
        Returns:
            List of loop entries
        """
        # Filter loops with metrics
        loops_with_metrics = []
        for loop in self.loop_cache["loops"]:
            loop_id = loop.get("id")
            if loop_id in self.loop_metrics and self.loop_metrics[loop_id]["executions"] > 0:
                loops_with_metrics.append(loop)
        
        # Sort by average profit
        sorted_loops = sorted(
            loops_with_metrics,
            key=lambda x: self.loop_metrics[x.get("id")]["avg_profit"],
            reverse=True
        )
        
        return sorted_loops[:limit]
    
    def prune_inactive_loops(self, max_age_days: int = 7) -> int:
        """
        Remove loops that haven't been active for a certain period.
        
        Args:
            max_age_days: Maximum age in days for inactive loops
            
        Returns:
            Number of loops removed
        """
        max_age_seconds = max_age_days * 24 * 60 * 60
        current_time = time.time()
        
        original_length = len(self.loop_cache["loops"])
        
        # Keep loops that are either recent or have been executed recently
        active_loops = []
        for loop in self.loop_cache["loops"]:
            loop_id = loop.get("id")
            discovered_at = loop.get("discovered_at", 0)
            
            # Check if loop has been executed recently
            recently_executed = False
            if loop_id in self.loop_metrics:
                last_execution = self.loop_metrics[loop_id].get("last_execution", 0)
                if current_time - last_execution < max_age_seconds:
                    recently_executed = True
            
            # Keep if recently discovered or executed
            if current_time - discovered_at < max_age_seconds or recently_executed:
                active_loops.append(loop)
        
        self.loop_cache["loops"] = active_loops
        
        # Save if loops were removed
        if len(active_loops) < original_length:
            self.save_cache()
        
        return original_length - len(active_loops)
    
    def clear_cache(self) -> None:
        """Clear the loop cache."""
        self.loop_cache = {"loops": [], "version": 1, "last_updated": time.time()}
        self.loop_metrics = {}
        self.save_cache()
