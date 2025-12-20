import numpy as np
from numba import njit

@njit
def find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])
    return parent[i]

@njit
def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
        return True
    return False

@njit
def process_batch(sorted_edges, start_idx, batch_size, parent, rank, state_buffer):
    count = 0
    limit = min(start_idx + batch_size, len(sorted_edges))
    
    for i in range(start_idx, limit):
        u, v, _ = sorted_edges[i]
        if union(parent, rank, int(u), int(v)):
            status = 2 # MST
        else:
            status = 1 # Rejected
        state_buffer[i*2] = status
        state_buffer[i*2 + 1] = status
        count += 1
        
    return count