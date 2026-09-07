#!/usr/bin/env python3
"""
Test Dijkstra path planning on the cleaned hybrid map.
Prints waypoints from START to GOAL.
"""
import json,os
import math
import heapq

def load_graph():
    try:
        with open(os.path.expanduser('~/gen_ws/src/amcc_nav/maps/hybrid_map_clean.json')) as f:
            return json.load(f)
    except FileNotFoundError:
        print('ERROR: /tmp/hybrid_map_clean.json not found')
        print('Run graph_pruner first!')
        return None

def nearest_vertex(verts, x, y):
    best, best_d = 0, float('inf')
    for v in verts:
        d = math.sqrt((v['x']-x)**2 + (v['y']-y)**2)
        if d < best_d:
            best_d = d
            best   = v['id']
    return best, best_d

def dijkstra(verts, edges, start_id, goal_id):
    n   = len(verts)
    INF = float('inf')
    dist = [INF] * n
    prev = [-1]  * n
    dist[start_id] = 0.0

    adj = {i: [] for i in range(n)}
    for e in edges:
        adj[e['u']].append((e['v'], e['weight']))
        adj[e['v']].append((e['u'], e['weight']))

    pq      = [(0.0, start_id)]
    visited = [False] * n

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]: continue
        visited[u] = True
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    # Reconstruct path
    path, cur = [], goal_id
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    # If path doesn't start at start_id → no path found
    if not path or path[0] != start_id:
        return [], INF

    return path, dist[goal_id]

def main():
    data = load_graph()
    if data is None:
        return

    verts = data['vertices']
    edges = data['edges']

    print('\n' + '='*50)
    print('HYBRID MAP PATH PLANNER TEST')
    print('='*50)
    print(f'Graph: {len(verts)} vertices, {len(edges)} edges')

    # Print all vertices so user can pick start/goal
    print('\nAvailable vertices:')
    for v in verts:
        print(f"  v{v['id']:2d} | ({v['x']:7.2f}, {v['y']:7.2f})")

    print('\nEnter START vertex id: ', end='')
    try:
        s_id = int(input())
        print('Enter GOAL  vertex id: ', end='')
        g_id = int(input())
    except ValueError:
        print('Invalid input — using v0 → v10 as default')
        s_id, g_id = 0, 10

    # Validate ids
    ids = [v['id'] for v in verts]
    if s_id not in ids or g_id not in ids:
        print(f'ERROR: vertex id must be one of {ids}')
        return

    print(f'\nPlanning from v{s_id} to v{g_id}...')

    path_ids, total_dist = dijkstra(verts, edges, s_id, g_id)

    if not path_ids:
        print('❌ No path found — graph may be disconnected')
        return

    print(f'\n✅ Path found: {len(path_ids)} waypoints')
    print(f'   Total distance: {total_dist:.2f}m')
    print('\nWaypoints:')
    print(f"  {'Step':>4} | {'Vertex':>6} | {'X':>8} | {'Y':>8}")
    print('  ' + '-'*36)
    for i, vid in enumerate(path_ids):
        v = verts[vid]
        print(f"  {i+1:>4} | v{vid:>5} | {v['x']:>8.2f} | {v['y']:>8.2f}")

    print(f'\nStart : v{s_id} ({verts[s_id]["x"]:.2f}, {verts[s_id]["y"]:.2f})')
    print(f'Goal  : v{g_id} ({verts[g_id]["x"]:.2f}, {verts[g_id]["y"]:.2f})')
    print(f'Hops  : {len(path_ids)-1}')
    print(f'Dist  : {total_dist:.2f}m')

if __name__ == '__main__':
    main()