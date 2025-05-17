import numpy as np
import heapq
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger("graph_algorithms")

@dataclass
class GraphNode:
    """Represents a node in the exchange rate graph."""
    asset: str
    
    def __hash__(self):
        return hash(self.asset)
    
    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.asset == other.asset

@dataclass
class GraphEdge:
    """Represents an edge in the exchange rate graph."""
    from_node: GraphNode
    to_node: GraphNode
    exchange: str
    rate: float
    liquidity: float
    timestamp: float
    
    def __hash__(self):
        return hash((self.from_node, self.to_node, self.exchange))
    
    def __eq__(self, other):
        if not isinstance(other, GraphEdge):
            return False
        return (self.from_node == other.from_node and 
                self.to_node == other.to_node and 
                self.exchange == other.exchange)

class ArbitrageGraph:
    """Graph representation for arbitrage detection algorithms."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        self.adjacency_list: Dict[str, List[GraphEdge]] = {}
        self.dirty_nodes: Set[str] = set()
        
    def get_or_create_node(self, asset: str) -> GraphNode:
        """Get an existing node or create a new one."""
        if asset not in self.nodes:
            self.nodes[asset] = GraphNode(asset)
        return self.nodes[asset]
    
    def add_edge(self, from_asset: str, to_asset: str, exchange: str, 
                 rate: float, liquidity: float) -> bool:
        """
        Add or update an edge in the graph.
        Returns True if the edge was added or updated significantly.
        """
        # Get or create nodes
        from_node = self.get_or_create_node(from_asset)
        to_node = self.get_or_create_node(to_asset)
        
        # Create edge key
        key = (from_asset, to_asset, exchange)
        
        # Check if this is an update to an existing edge
        is_update = key in self.edges
        
        if is_update:
            old_edge = self.edges[key]
            rate_change = abs(rate - old_edge.rate) / old_edge.rate
            
            # If rate change is minimal, don't update
            if rate_change < 0.0001:  # 0.01% change threshold
                return False
        
        # Create new edge
        edge = GraphEdge(
            from_node=from_node,
            to_node=to_node,
            exchange=exchange,
            rate=rate,
            liquidity=liquidity,
            timestamp=time.time()
        )
        
        # Update the graph
        self.edges[key] = edge
        
        # Update adjacency list
        if from_asset not in self.adjacency_list:
            self.adjacency_list[from_asset] = []
        
        # Replace existing edge or add new one
        if is_update:
            for i, existing_edge in enumerate(self.adjacency_list[from_asset]):
                if (existing_edge.to_node.asset == to_asset and 
                    existing_edge.exchange == exchange):
                    self.adjacency_list[from_asset][i] = edge
                    break
        else:
            self.adjacency_list[from_asset].append(edge)
        
        # Mark nodes as dirty for incremental processing
        self.dirty_nodes.add(from_asset)
        self.dirty_nodes.add(to_asset)
        
        return True
    
    def get_subgraph(self, center_nodes: Set[str], radius: int = 2) -> Set[str]:
        """
        Get a subgraph containing the specified nodes and their neighbors 
        up to a certain radius.
        """
        subgraph = set(center_nodes)
        frontier = set(center_nodes)
        
        for _ in range(radius):
            new_frontier = set()
            
            for node in frontier:
                # Add outgoing neighbors
                if node in self.adjacency_list:
                    for edge in self.adjacency_list[node]:
                        new_frontier.add(edge.to_node.asset)
            
            # Add new nodes to subgraph and update frontier
            subgraph.update(new_frontier)
            frontier = new_frontier
            
            # If no new nodes were added, we can stop early
            if not frontier:
                break
        
        return subgraph
    
    def clear_dirty_flags(self):
        """Clear the set of dirty nodes."""
        self.dirty_nodes.clear()
    
    def to_weighted_adjacency_matrix(self, base_asset: str = None) -> Tuple[np.ndarray, List[str]]:
        """
        Convert the graph to a weighted adjacency matrix.
        
        Args:
            base_asset: Optional base asset to use for reweighting
            
        Returns:
            Tuple of (adjacency matrix, list of node labels)
        """
        # Get list of all nodes
        node_labels = list(self.nodes.keys())
        n = len(node_labels)
        
        # Create mapping from node label to index
        node_to_idx = {label: i for i, label in enumerate(node_labels)}
        
        # Initialize adjacency matrix with infinity
        adj_matrix = np.full((n, n), np.inf)
        
        # Set diagonal to zero
        np.fill_diagonal(adj_matrix, 0)
        
        # Fill in edge weights
        for from_asset, edges in self.adjacency_list.items():
            from_idx = node_to_idx[from_asset]
            
            for edge in edges:
                to_idx = node_to_idx[edge.to_node.asset]
                
                # Use negative log of rate as weight for shortest path algorithms
                # This converts multiplication to addition and finds maximum product
                weight = -np.log(edge.rate)
                
                # Use the minimum weight if there are multiple edges between nodes
                adj_matrix[from_idx, to_idx] = min(adj_matrix[from_idx, to_idx], weight)
        
        return adj_matrix, node_labels


class JohnsonAlgorithm:
    """
    Implementation of Johnson's algorithm for finding negative cycles in a graph.
    This is used for detecting arbitrage opportunities.
    """
    
    @staticmethod
    def find_negative_cycles(graph: ArbitrageGraph, 
                             subgraph: Set[str] = None,
                             base_asset: str = None) -> List[Dict]:
        """
        Find negative cycles in the graph using Johnson's algorithm.
        
        Args:
            graph: The arbitrage graph
            subgraph: Optional set of nodes to restrict the search to
            base_asset: Optional base asset to use for reweighting
            
        Returns:
            List of negative cycles (arbitrage opportunities)
        """
        start_time = time.time()
        
        # If subgraph is specified, filter the graph
        if subgraph:
            # Create a filtered adjacency list
            filtered_adj_list = {}
            for from_asset, edges in graph.adjacency_list.items():
                if from_asset in subgraph:
                    filtered_edges = [e for e in edges if e.to_node.asset in subgraph]
                    if filtered_edges:
                        filtered_adj_list[from_asset] = filtered_edges
            
            # Use the filtered adjacency list for processing
            adj_list = filtered_adj_list
            node_set = subgraph
        else:
            # Use the full graph
            adj_list = graph.adjacency_list
            node_set = set(graph.nodes.keys())
        
        # Convert to list for indexing
        node_list = list(node_set)
        n = len(node_list)
        
        # Create mapping from node label to index
        node_to_idx = {label: i for i, label in enumerate(node_list)}
        
        # Initialize distance matrix with infinity
        dist_matrix = np.full((n, n), np.inf)
        
        # Set diagonal to zero
        np.fill_diagonal(dist_matrix, 0)
        
        # Initialize predecessor matrix
        pred_matrix = np.full((n, n), -1, dtype=int)
        
        # Fill in edge weights
        for from_asset, edges in adj_list.items():
            if from_asset not in node_to_idx:
                continue
                
            from_idx = node_to_idx[from_asset]
            
            for edge in edges:
                to_asset = edge.to_node.asset
                if to_asset not in node_to_idx:
                    continue
                    
                to_idx = node_to_idx[to_asset]
                
                # Use negative log of rate as weight for shortest path algorithms
                weight = -np.log(edge.rate)
                
                # Use the minimum weight if there are multiple edges between nodes
                if weight < dist_matrix[from_idx, to_idx]:
                    dist_matrix[from_idx, to_idx] = weight
                    pred_matrix[from_idx, to_idx] = from_idx
        
        # Find negative cycles
        negative_cycles = []
        
        # Run Bellman-Ford from each node to find potential negative cycles
        for source_idx in range(n):
            # Skip if this node has no outgoing edges
            if np.all(dist_matrix[source_idx] == np.inf):
                continue
                
            # Initialize distance and predecessor arrays for Bellman-Ford
            dist = np.full(n, np.inf)
            dist[source_idx] = 0
            pred = np.full(n, -1, dtype=int)
            
            # Relax edges |V| - 1 times
            for _ in range(n - 1):
                for u in range(n):
                    for v in range(n):
                        if dist_matrix[u, v] != np.inf and dist[u] != np.inf:
                            if dist[v] > dist[u] + dist_matrix[u, v]:
                                dist[v] = dist[u] + dist_matrix[u, v]
                                pred[v] = u
            
            # Check for negative cycles
            for u in range(n):
                for v in range(n):
                    if (dist_matrix[u, v] != np.inf and dist[u] != np.inf and 
                        dist[v] > dist[u] + dist_matrix[u, v]):
                        # Negative cycle detected
                        # Trace back to find the cycle
                        cycle_nodes = []
                        visited = set()
                        
                        # Start from v
                        curr = v
                        while curr not in visited and len(cycle_nodes) < n:
                            visited.add(curr)
                            cycle_nodes.append(curr)
                            curr = pred[curr]
                        
                        # Find where the cycle starts
                        if curr in cycle_nodes:
                            start_idx = cycle_nodes.index(curr)
                            cycle = cycle_nodes[start_idx:]
                            
                            # Convert indices back to asset names
                            cycle_assets = [node_list[idx] for idx in cycle]
                            
                            # Calculate profit
                            profit = 1.0
                            exchanges = []
                            
                            for i in range(len(cycle_assets)):
                                from_asset = cycle_assets[i]
                                to_asset = cycle_assets[(i + 1) % len(cycle_assets)]
                                
                                # Find the edge with the best rate
                                best_rate = 0
                                best_exchange = None
                                
                                if from_asset in adj_list:
                                    for edge in adj_list[from_asset]:
                                        if edge.to_node.asset == to_asset and edge.rate > best_rate:
                                            best_rate = edge.rate
                                            best_exchange = edge.exchange
                                
                                if best_rate > 0:
                                    profit *= best_rate
                                    exchanges.append(best_exchange)
                            
                            # Calculate profit percentage
                            profit_pct = (profit - 1) * 100
                            
                            # Add to results if profit is significant
                            if profit_pct > 0.1:  # 0.1% minimum profit
                                # Format as trading pairs
                                trading_pairs = []
                                for i in range(len(cycle_assets)):
                                    from_asset = cycle_assets[i]
                                    to_asset = cycle_assets[(i + 1) % len(cycle_assets)]
                                    trading_pairs.append(f"{from_asset}/{to_asset}")
                                
                                negative_cycles.append({
                                    'loop': trading_pairs,
                                    'profit_percent': profit_pct,
                                    'exchanges': exchanges,
                                    'assets': cycle_assets
                                })
        
        # Log performance
        duration = time.time() - start_time
        logger.debug(f"Johnson's algorithm found {len(negative_cycles)} cycles in {duration:.3f} seconds")
        
        return negative_cycles


class BellmanFordAlgorithm:
    """
    Implementation of the Bellman-Ford algorithm for finding negative cycles in a graph.
    This is used as a fallback for detecting arbitrage opportunities.
    """
    
    @staticmethod
    def find_negative_cycles(graph: ArbitrageGraph, 
                             subgraph: Set[str] = None,
                             base_asset: str = None) -> List[Dict]:
        """
        Find negative cycles in the graph using the Bellman-Ford algorithm.
        
        Args:
            graph: The arbitrage graph
            subgraph: Optional set of nodes to restrict the search to
            base_asset: Optional base asset to use for reweighting
            
        Returns:
            List of negative cycles (arbitrage opportunities)
        """
        start_time = time.time()
        
        # If no base asset is specified, use a common one
        if not base_asset:
            base_asset = "USDT"
        
        # If base asset is not in the graph, use the first node
        if base_asset not in graph.nodes:
            if graph.nodes:
                base_asset = next(iter(graph.nodes.keys()))
            else:
                return []  # Empty graph
        
        # If subgraph is specified, filter the graph
        if subgraph:
            # Create a filtered adjacency list
            filtered_adj_list = {}
            for from_asset, edges in graph.adjacency_list.items():
                if from_asset in subgraph:
                    filtered_edges = [e for e in edges if e.to_node.asset in subgraph]
                    if filtered_edges:
                        filtered_adj_list[from_asset] = filtered_edges
            
            # Use the filtered adjacency list for processing
            adj_list = filtered_adj_list
            node_set = subgraph
        else:
            # Use the full graph
            adj_list = graph.adjacency_list
            node_set = set(graph.nodes.keys())
        
        # Convert to list for indexing
        node_list = list(node_set)
        n = len(node_list)
        
        # Create mapping from node label to index
        node_to_idx = {label: i for i, label in enumerate(node_list)}
        
        # Create list of edges for Bellman-Ford
        edges = []
        for from_asset, edge_list in adj_list.items():
            if from_asset not in node_to_idx:
                continue
                
            from_idx = node_to_idx[from_asset]
            
            for edge in edge_list:
                to_asset = edge.to_node.asset
                if to_asset not in node_to_idx:
                    continue
                    
                to_idx = node_to_idx[to_asset]
                
                # Use negative log of rate as weight
                weight = -np.log(edge.rate)
                
                edges.append((from_idx, to_idx, weight, edge.exchange))
        
        # Initialize distance and predecessor arrays
        dist = np.full(n, np.inf)
        pred = np.full(n, -1, dtype=int)
        
        # Set distance to base asset to 0
        if base_asset in node_to_idx:
            source_idx = node_to_idx[base_asset]
            dist[source_idx] = 0
        
        # Relax edges |V| - 1 times
        for _ in range(n - 1):
            for u, v, w, _ in edges:
                if dist[u] != np.inf and dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    pred[v] = u
        
        # Check for negative cycles
        negative_cycles = []
        
        for u, v, w, exchange in edges:
            if dist[u] != np.inf and dist[v] > dist[u] + w:
                # Negative cycle detected
                # Trace back to find the cycle
                cycle_nodes = []
                visited = set()
                
                # Start from v
                curr = v
                while curr not in visited and len(cycle_nodes) < n:
                    visited.add(curr)
                    cycle_nodes.append(curr)
                    curr = pred[curr]
                
                # Find where the cycle starts
                if curr in cycle_nodes:
                    start_idx = cycle_nodes.index(curr)
                    cycle = cycle_nodes[start_idx:]
                    
                    # Convert indices back to asset names
                    cycle_assets = [node_list[idx] for idx in cycle]
                    
                    # Calculate profit
                    profit = 1.0
                    exchanges = []
                    
                    for i in range(len(cycle_assets)):
                        from_asset = cycle_assets[i]
                        to_asset = cycle_assets[(i + 1) % len(cycle_assets)]
                        
                        # Find the edge with the best rate
                        best_rate = 0
                        best_exchange = None
                        
                        if from_asset in adj_list:
                            for edge in adj_list[from_asset]:
                                if edge.to_node.asset == to_asset and edge.rate > best_rate:
                                    best_rate = edge.rate
                                    best_exchange = edge.exchange
                        
                        if best_rate > 0:
                            profit *= best_rate
                            exchanges.append(best_exchange)
                    
                    # Calculate profit percentage
                    profit_pct = (profit - 1) * 100
                    
                    # Add to results if profit is significant
                    if profit_pct > 0.1:  # 0.1% minimum profit
                        # Format as trading pairs
                        trading_pairs = []
                        for i in range(len(cycle_assets)):
                            from_asset = cycle_assets[i]
                            to_asset = cycle_assets[(i + 1) % len(cycle_assets)]
                            trading_pairs.append(f"{from_asset}/{to_asset}")
                        
                        negative_cycles.append({
                            'loop': trading_pairs,
                            'profit_percent': profit_pct,
                            'exchanges': exchanges,
                            'assets': cycle_assets
                        })
        
        # Log performance
        duration = time.time() - start_time
        logger.debug(f"Bellman-Ford algorithm found {len(negative_cycles)} cycles in {duration:.3f} seconds")
        
        return negative_cycles


class YenKShortestPaths:
    """
    Implementation of Yen's K-shortest paths algorithm.
    This is used for finding multiple profitable paths between assets.
    """
    
    @staticmethod
    def find_k_shortest_paths(graph: ArbitrageGraph, 
                              source: str, 
                              target: str, 
                              k: int = 5) -> List[Dict]:
        """
        Find k shortest paths from source to target using Yen's algorithm.
        
        Args:
            graph: The arbitrage graph
            source: Source asset
            target: Target asset
            k: Number of paths to find
            
        Returns:
            List of paths sorted by total weight
        """
        if source not in graph.nodes or target not in graph.nodes:
            return []
        
        # Get adjacency list
        adj_list = graph.adjacency_list
        
        # Initialize result list
        A = []
        
        # Find the shortest path using Dijkstra's algorithm
        shortest_path = YenKShortestPaths._dijkstra(graph, source, target)
        
        if not shortest_path:
            return []
        
        A.append(shortest_path)
        
        # Initialize potential k-shortest paths
        B = []
        
        # Find k-1 more paths
        for k_idx in range(1, k):
            # The previous k-shortest path
            prev_path = A[-1]
            
            # For each node in the previous path except the last
            for i in range(len(prev_path['assets']) - 1):
                # Spur node is the i-th node in the previous path
                spur_node = prev_path['assets'][i]
                
                # Root path is the path from source to spur node
                root_path = {
                    'assets': prev_path['assets'][:i+1],
                    'exchanges': prev_path['exchanges'][:i],
                    'rates': prev_path['rates'][:i],
                    'total_rate': 1.0
                }
                
                # Update root path total rate
                for rate in root_path['rates']:
                    root_path['total_rate'] *= rate
                
                # Remove edges that are part of previously found k-shortest paths
                # if they share the same root path
                removed_edges = []
                
                for path in A:
                    if (len(path['assets']) > i + 1 and 
                        path['assets'][:i+1] == root_path['assets']):
                        # Remove the edge from spur node to the next node in this path
                        u = path['assets'][i]
                        v = path['assets'][i+1]
                        
                        if u in adj_list:
                            for edge in adj_list[u]:
                                if edge.to_node.asset == v:
                                    removed_edges.append((u, v, edge))
                
                # Temporarily remove edges
                for u, v, edge in removed_edges:
                    adj_list[u].remove(edge)
                
                # Find the spur path from spur node to target
                spur_path = YenKShortestPaths._dijkstra(graph, spur_node, target)
                
                # Restore removed edges
                for u, v, edge in removed_edges:
                    adj_list[u].append(edge)
                
                # If a spur path was found
                if spur_path:
                    # Concatenate root path and spur path
                    total_path = {
                        'assets': root_path['assets'] + spur_path['assets'][1:],
                        'exchanges': root_path['exchanges'] + spur_path['exchanges'],
                        'rates': root_path['rates'] + spur_path['rates'],
                        'total_rate': root_path['total_rate'] * spur_path['total_rate']
                    }
                    
                    # Add to potential k-shortest paths if not already there
                    if not any(YenKShortestPaths._paths_equal(total_path, p) for p in B):
                        B.append(total_path)
            
            # If no more paths are available
            if not B:
                break
            
            # Sort potential paths by total rate (descending)
            B.sort(key=lambda x: x['total_rate'], reverse=True)
            
            # Add the best path to the result
            A.append(B[0])
            B.pop(0)
        
        # Convert to the expected format
        result = []
        for path in A:
            # Calculate profit percentage
            profit_pct = (path['total_rate'] - 1) * 100
            
            # Format as trading pairs
            trading_pairs = []
            for i in range(len(path['assets']) - 1):
                from_asset = path['assets'][i]
                to_asset = path['assets'][i+1]
                trading_pairs.append(f"{from_asset}/{to_asset}")
            
            result.append({
                'loop': trading_pairs,
                'profit_percent': profit_pct,
                'exchanges': path['exchanges'],
                'assets': path['assets']
            })
        
        return result
    
    @staticmethod
    def _dijkstra(graph: ArbitrageGraph, source: str, target: str) -> Optional[Dict]:
        """
        Find the shortest path from source to target using Dijkstra's algorithm.
        
        Args:
            graph: The arbitrage graph
            source: Source asset
            target: Target asset
            
        Returns:
            Dictionary with path details or None if no path exists
        """
        # Get adjacency list
        adj_list = graph.adjacency_list
        
        # Initialize distances with infinity
        dist = {node: float('inf') for node in graph.nodes}
        dist[source] = 0
        
        # Initialize predecessors
        pred = {node: None for node in graph.nodes}
        
        # Initialize rates
        rates = {node: 1.0 for node in graph.nodes}
        
        # Initialize exchanges
        exchanges = {node: None for node in graph.nodes}
        
        # Priority queue for Dijkstra
        pq = [(0, source)]
        
        # Set of visited nodes
        visited = set()
        
        while pq:
            # Get node with minimum distance
            curr_dist, curr_node = heapq.heappop(pq)
            
            # If already visited, skip
            if curr_node in visited:
                continue
            
            # Mark as visited
            visited.add(curr_node)
            
            # If target is reached, stop
            if curr_node == target:
                break
            
            # If no outgoing edges, skip
            if curr_node not in adj_list:
                continue
            
            # Explore neighbors
            for edge in adj_list[curr_node]:
                neighbor = edge.to_node.asset
                
                # Use negative log of rate as weight
                weight = -np.log(edge.rate)
                
                # If a shorter path is found
                if dist[neighbor] > dist[curr_node] + weight:
                    dist[neighbor] = dist[curr_node] + weight
                    pred[neighbor] = curr_node
                    rates[neighbor] = edge.rate
                    exchanges[neighbor] = edge.exchange
                    
                    # Add to priority queue
                    heapq.heappush(pq, (dist[neighbor], neighbor))
        
        # If target is not reachable
        if pred[target] is None:
            return None
        
        # Reconstruct the path
        path = []
        path_exchanges = []
        path_rates = []
        
        curr = target
        while curr != source:
            path.append(curr)
            if exchanges[curr]:
                path_exchanges.append(exchanges[curr])
            if rates[curr]:
                path_rates.append(rates[curr])
            curr = pred[curr]
        
        path.append(source)
        
        # Reverse to get path from source to target
        path.reverse()
        path_exchanges.reverse()
        path_rates.reverse()
        
        # Calculate total rate
        total_rate = 1.0
        for rate in path_rates:
            total_rate *= rate
        
        return {
            'assets': path,
            'exchanges': path_exchanges,
            'rates': path_rates,
            'total_rate': total_rate
        }
    
    @staticmethod
    def _paths_equal(path1: Dict, path2: Dict) -> bool:
        """Check if two paths are equal."""
        return path1['assets'] == path2['assets']
