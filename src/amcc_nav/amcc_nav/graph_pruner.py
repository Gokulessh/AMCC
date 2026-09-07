#!/usr/bin/env python3
"""
Loads hybrid_map.json, removes edges that cross walls,
saves cleaned graph to hybrid_map_clean.json
Run ONCE after teleop is complete.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import json, math,os
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
class GraphPruner(Node):
    def __init__(self):
        super().__init__('graph_pruner')
        self.occupancy_grid = None
        self.inflated_map   = None
        self.inflation_radius   = 0.3
        self.wall_check_samples = 50
        self.max_edge_length    = 7.0

        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos)        
        self.get_logger().info('Waiting for map...')

    def map_callback(self, msg):
        if self.inflated_map is not None:
            return  # already processed
        self.occupancy_grid = msg
        self._build_inflated_map(msg)
        self.get_logger().info('Map received — pruning graph...')
        self.prune_and_save()

    def _build_inflated_map(self, msg):
        info = msg.info
        res  = info.resolution
        w    = info.width
        h    = info.height
        raw  = np.array(msg.data, dtype=np.int8).reshape(h, w)
        occupied = (raw > 50) | (raw < 0)
        inflation_cells = int(math.ceil(self.inflation_radius / res))
        inflated = occupied.copy()
        occ_rows, occ_cols = np.where(occupied)
        for r, c in zip(occ_rows, occ_cols):
            r_min = max(0,   r - inflation_cells)
            r_max = min(h-1, r + inflation_cells)
            c_min = max(0,   c - inflation_cells)
            c_max = min(w-1, c + inflation_cells)
            inflated[r_min:r_max+1, c_min:c_max+1] = True
        self.inflated_map = inflated

    def _is_path_free(self, x1, y1, x2, y2):
        info     = self.occupancy_grid.info
        res      = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        width    = info.width
        height   = info.height

        for i in range(self.wall_check_samples + 1):
            t  = i / self.wall_check_samples
            x  = x1 + t * (x2 - x1)
            y  = y1 + t * (y2 - y1)
            cx = int((x - origin_x) / res)
            cy = int((y - origin_y) / res)
            if cx < 0 or cy < 0 or cx >= width or cy >= height:
                return False
            if self.inflated_map[cy, cx]:
                return False
        return True

    def prune_and_save(self):
        # READ raw graph ← was reading clean by mistake
        with open(os.path.expanduser(
                '~/gen_ws/src/amcc_nav/maps/hybrid_map.json'), 'r') as f:
            data = json.load(f)

        verts = data['vertices']
        edges = data['edges']

        self.get_logger().info(
            f'Before pruning: {len(verts)} vertices, {len(edges)} edges')

        clean_edges = []
        for e in edges:
            u = verts[e['u']]
            v = verts[e['v']]
            dist = math.sqrt((u['x']-v['x'])**2 + (u['y']-v['y'])**2)

            if dist > self.max_edge_length:
                self.get_logger().warn(
                    f"Edge {e['u']}→{e['v']} REMOVED: too long ({dist:.2f}m)")
                continue

            if not self._is_path_free(u['x'], u['y'], v['x'], v['y']):
                self.get_logger().warn(
                    f"Edge {e['u']}→{e['v']} REMOVED: crosses wall")
                continue

            clean_edges.append(e)
            self.get_logger().info(
                f"Edge {e['u']}→{e['v']} KEPT ({dist:.2f}m)")

        connected_ids = set()
        for e in clean_edges:
            connected_ids.add(e['u'])
            connected_ids.add(e['v'])

        isolated = [v['id'] for v in verts if v['id'] not in connected_ids]
        if isolated:
            self.get_logger().warn(f'Isolated vertices: {isolated}')

        clean_data = {'vertices': verts, 'edges': clean_edges}

        # SAVE to permanent path ← was saving to /tmp by mistake
        save_path = os.path.expanduser(
            '~/gen_ws/src/amcc_nav/maps/hybrid_map_clean.json')
        with open(save_path, 'w') as f:
            json.dump(clean_data, f, indent=2)

        self.get_logger().info(
            f'After pruning: {len(verts)} vertices, '
            f'{len(clean_edges)} edges')
        self.get_logger().info(f'Saved to {save_path}')

        # Connectivity check
        adj = {v['id']: [] for v in verts}
        for e in clean_edges:
            adj[e['u']].append(e['v'])
            adj[e['v']].append(e['u'])

        visited, queue = set(), [0]
        while queue:
            n = queue.pop(0)
            if n in visited: continue
            visited.add(n)
            queue.extend(adj[n])

        print(f'\nConnected: {len(visited)}/{len(verts)}')
        if len(visited) == len(verts):
            print('✅ Graph fully connected')
        else:
            not_visited = [v['id'] for v in verts
                        if v['id'] not in visited]
            print(f'❌ Disconnected vertices: {not_visited}')

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GraphPruner())

if __name__ == '__main__':
    main()