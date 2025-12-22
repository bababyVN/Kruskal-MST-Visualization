import numpy as np
from numba import njit

@njit
def find(parent, i):
    path = []
    root = i
    while parent[root] != root:
        path.append(root)
        root = parent[root]
    for node in path:
        parent[node] = root
    return root

@njit
def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
            return True, root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
            return True, root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
            return True, root_x
    return False, root_x

@njit
def get_all_roots(parent):
    roots = np.empty_like(parent)
    for i in range(len(parent)):
        roots[i] = find(parent, i)
    return roots

@njit
def get_node_statuses(parent, rank):
    status = np.zeros(len(parent), dtype=np.float32)
    for i in range(len(parent)):
        if parent[i] != i or rank[i] > 0:
            status[i] = 1.0
    return status

@njit
def process_batch(sorted_edges, start_idx, limit, parent, rank, state_data, current_mst, target_mst):
    processed_count = 0
    mst_added_count = 0
    total_weight = 0.0
    
    for i in range(limit):
        if start_idx + i >= len(sorted_edges):
            break
            
        u, v, w = sorted_edges[start_idx + i]
        is_merged, _ = union(parent, rank, int(u), int(v))
        
        idx_in_buffer = (start_idx + i) * 2
        
        if is_merged:
            state_data[idx_in_buffer] = 2.0
            state_data[idx_in_buffer+1] = 2.0 
            mst_added_count += 1
            total_weight += w
        else:
            state_data[idx_in_buffer] = 1.0
            state_data[idx_in_buffer+1] = 1.0
            
        processed_count += 1
        
        if current_mst + mst_added_count >= target_mst:
            break
            
    return processed_count, mst_added_count, total_weight

@njit
def fast_forward_dsu(sorted_edges, limit, parent, rank):
    mst_added_count = 0
    total_weight = 0.0
    
    for i in range(limit):
        u, v, w = sorted_edges[i]
        is_merged, _ = union(parent, rank, int(u), int(v))
        if is_merged:
            mst_added_count += 1
            total_weight += w
            
    return mst_added_count, total_weight

def prepare_data(n_vertices, n_edges):
    print(f"Generating Large Scale Data: {n_vertices} Vertices, {n_edges} Edges...")
    
    # 1. Generate Nodes
    scale_factor = 2000.0 
    radius = np.sqrt(np.random.uniform(0, 1, n_vertices)) * scale_factor
    angle = np.random.uniform(0, 2 * np.pi, n_vertices)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    
    nodes = np.column_stack((x, y)).astype('f4')
    
    # 2. Generate Random Edges
    u = np.random.randint(0, n_vertices, n_edges)
    v = np.random.randint(0, n_vertices, n_edges)
    weights = np.random.uniform(1, 100, n_edges)
    
    # Filter self-loops
    mask = u != v
    u, v, weights = u[mask], v[mask], weights[mask]
    
    # Create raw edge array
    raw_edges = np.column_stack((u, v, weights)).astype(np.float64)
    
    # 3. SORT EDGES (Critical for Kruskal's)
    # We must sort by weight (column 2)
    sorted_indices = np.argsort(raw_edges[:, 2])
    sorted_edges = raw_edges[sorted_indices]
    
    # 4. BUILD GEOMETRY (Critical for Rendering)
    # We need to map edge indices -> node coordinates
    # shape: (N_edges * 2, 2)
    
    # Re-extract sorted u and v indices as integers
    u_sorted = sorted_edges[:, 0].astype(int)
    v_sorted = sorted_edges[:, 1].astype(int)
    
    # Fetch coordinates using fancy indexing (fast)
    start_points = nodes[u_sorted]
    end_points = nodes[v_sorted]
    
    # Interleave them: [Start0, End0, Start1, End1, ...]
    n_lines = len(sorted_edges)
    line_geometry = np.empty((n_lines * 2, 2), dtype='f4')
    line_geometry[0::2] = start_points
    line_geometry[1::2] = end_points
    
    print("Data Generation Complete.")
    return sorted_edges, line_geometry, nodes