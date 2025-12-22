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
        is_merged, root_val = union(parent, rank, int(u), int(v))
        
        idx_in_buffer = (start_idx + i) * 2
        
        if is_merged:
            # Set to generic MST state initially (will be colored by refresh)
            # Using 10.0 + root immediately helps, but roots change!
            val = 10.0 + float(root_val)
            state_data[idx_in_buffer] = val
            state_data[idx_in_buffer+1] = val
            mst_added_count += 1
            total_weight += w
        else:
            state_data[idx_in_buffer] = 1.0
            state_data[idx_in_buffer+1] = 1.0
            
        processed_count += 1
        
        if current_mst + mst_added_count >= target_mst:
            break
            
    return processed_count, mst_added_count, total_weight

# --- NEW FUNCTION ---
@njit
def refresh_mst_colors(sorted_edges, limit, parent, state_data):
    """
    Re-calculates the root ID for ALL MST edges up to the current limit.
    This ensures edges change color when their tree merges with another tree.
    """
    for i in range(limit):
        idx = i * 2
        # Check if this edge is part of MST (Status >= 2.0)
        if state_data[idx] >= 2.0:
            u = int(sorted_edges[i, 0])
            # Find the CURRENT root of this edge's component
            current_root = find(parent, u)
            # Update the color ID
            val = 10.0 + float(current_root)
            state_data[idx] = val
            state_data[idx+1] = val

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
    
    scale_factor = 2000.0 
    radius = np.sqrt(np.random.uniform(0, 1, n_vertices)) * scale_factor
    angle = np.random.uniform(0, 2 * np.pi, n_vertices)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    
    vertices = np.column_stack((x, y)).astype('f4')
    
    u = np.random.randint(0, n_vertices, n_edges)
    v = np.random.randint(0, n_vertices, n_edges)
    weights = np.random.uniform(1, 100, n_edges)
    
    mask = u != v
    u, v, weights = u[mask], v[mask], weights[mask]
    
    edges = np.column_stack((u, v, weights)).astype(np.float64)
    
    sorted_indices = np.argsort(edges[:, 2])
    sorted_edges = edges[sorted_indices]
    
    u_sorted = sorted_edges[:, 0].astype(int)
    v_sorted = sorted_edges[:, 1].astype(int)
    start_points = vertices[u_sorted]
    end_points = vertices[v_sorted]
    
    n_lines = len(sorted_edges)
    line_geometry = np.empty((n_lines * 2, 2), dtype='f4')
    line_geometry[0::2] = start_points
    line_geometry[1::2] = end_points
    
    return sorted_edges, line_geometry, vertices