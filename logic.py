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
def process_single_edge(u, v, parent, rank):
    is_merged, new_root = union(parent, rank, int(u), int(v))
    return (2 if is_merged else 1), new_root

@njit
def process_batch(sorted_edges, start_idx, batch_size, parent, rank, state_buffer):
    """
    Returns: (edges_processed, edges_added_to_mst)
    """
    processed_count = 0
    mst_added_count = 0
    limit = min(start_idx + batch_size, len(sorted_edges))
    
    for i in range(start_idx, limit):
        u, v, _ = sorted_edges[i]
        is_merged, _ = union(parent, rank, int(u), int(v))
        
        if is_merged:
            status = 2 # MST
            mst_added_count += 1
        else:
            status = 1 # Rejected
            
        state_buffer[i*2] = status
        state_buffer[i*2 + 1] = status
        processed_count += 1
        
    return processed_count, mst_added_count

def prepare_data(n_vertices, n_edges):
    print(f"Generating Circular Data: {n_vertices} Vertices, {n_edges} Edges...")
    
    # --- CIRCULAR GENERATION (Polar Coordinates) ---
    # 1. Random Radius (sqrt for uniform distribution) and Angle
    radius = np.sqrt(np.random.uniform(0, 1, n_vertices)) * 0.9
    angle = np.random.uniform(0, 2 * np.pi, n_vertices)
    
    # 2. Convert to Cartesian
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    vertices = np.column_stack((x, y)).astype('f4')
    
    # 3. Random Edges
    u = np.random.randint(0, n_vertices, n_edges)
    v = np.random.randint(0, n_vertices, n_edges)
    weights = np.random.uniform(1, 100, n_edges)
    
    # Remove self-loops
    mask = u != v
    u, v, weights = u[mask], v[mask], weights[mask]
    edges = np.column_stack((u, v, weights))
    
    print("Sorting edges...")
    sorted_indices = np.argsort(edges[:, 2])
    sorted_edges = edges[sorted_indices]
    
    start_points = vertices[sorted_edges[:, 0].astype(int)]
    end_points = vertices[sorted_edges[:, 1].astype(int)]
    
    line_geometry = np.empty((len(sorted_edges) * 2, 2), dtype='f4')
    line_geometry[0::2] = start_points
    line_geometry[1::2] = end_points
    
    return sorted_edges, line_geometry, vertices