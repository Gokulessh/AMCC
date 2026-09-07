#!/usr/bin/env python3
"""
Benchmarks:
  1. Dijkstra on topological graph (hybrid map)
  2. A* on occupancy grid via Nav2 ComputePathToPose

Prints timing + path length comparison table.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                        QoSReliabilityPolicy)
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
import json, os, math, heapq, time
import numpy as np

GRAPH_PATH = os.path.expanduser(
    '~/gen_ws/src/amcc_nav/maps/hybrid_map_clean.json')

class PlannerBenchmark(Node):
    def __init__(self):
        super().__init__('planner_benchmark')

        # ── Load topo graph ─────────────────────────────────────────
        with open(GRAPH_PATH) as f:
            data = json.load(f)
        self.verts = data['vertices']
        self.edges = data['edges']
        self.get_logger().info(
            f'Graph: {len(self.verts)} vertices, '
            f'{len(self.edges)} edges')

        # ── Map subscriber ───────────────────────────────────────────
        self.occupancy_grid = None
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos)

        # ── Nav2 action client ───────────────────────────────────────
        self._nav2 = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose')

        self.get_logger().info('Waiting for map and Nav2...')

    # ─────────────────────────────────────────────
    # Map
    # ─────────────────────────────────────────────
    def map_callback(self, msg):
        if self.occupancy_grid is not None:
            return
        self.occupancy_grid = msg
        self.get_logger().info(
            f'Map received: '
            f'{msg.info.width}x{msg.info.height} cells')
        # Wait a moment then run
        self.create_timer(2.0, self._start)

    def _start(self):
        self.timer_start.cancel()
        self.run()

    # ─────────────────────────────────────────────
    # Topo planner — Dijkstra
    # ─────────────────────────────────────────────
    def nearest_vertex(self, x, y):
        best, best_d = 0, float('inf')
        for v in self.verts:
            d = math.sqrt((v['x']-x)**2 + (v['y']-y)**2)
            if d < best_d:
                best_d, best = d, v['id']
        return best

    def plan_topo(self, sx, sy, gx, gy):
        t0   = time.perf_counter()
        s_id = self.nearest_vertex(sx, sy)
        g_id = self.nearest_vertex(gx, gy)
        n    = len(self.verts)
        INF  = float('inf')
        dist = [INF] * n
        prev = [-1]  * n
        dist[s_id] = 0.0
        adj  = {i: [] for i in range(n)}
        for e in self.edges:
            adj[e['u']].append((e['v'], e['weight']))
            adj[e['v']].append((e['u'], e['weight']))
        pq      = [(0.0, s_id)]
        visited = [False] * n
        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]: continue
            visited[u] = True
            for v, w in adj[u]:
                if dist[u]+w < dist[v]:
                    dist[v] = dist[u]+w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
        path, cur = [], g_id
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        t1 = time.perf_counter()
        path_len = dist[g_id] if dist[g_id] < INF else 0.0
        return (t1-t0)*1000, path_len, len(path)

    # ─────────────────────────────────────────────
    # Grid planner — Nav2 A*
    # ─────────────────────────────────────────────
    def plan_nav2(self, sx, sy, gx, gy):
        """Call Nav2 ComputePathToPose and time it."""
        if not self._nav2.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                'Nav2 compute_path_to_pose not available!')
            return None, None, None

        def make_pose(x, y):
            p = PoseStamped()
            p.header.frame_id    = 'map'
            p.header.stamp       = self.get_clock().now().to_msg()
            p.pose.position.x    = x
            p.pose.position.y    = y
            p.pose.orientation.w = 1.0
            return p

        goal_msg       = ComputePathToPose.Goal()
        goal_msg.start = make_pose(sx, sy)
        goal_msg.goal  = make_pose(gx, gy)
        goal_msg.use_start = True

        t0     = time.perf_counter()
        future = self._nav2.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=15.0)

        if not future.done():
            return None, None, None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 goal rejected!')
            return None, None, None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=15.0)
        t1 = time.perf_counter()

        if not result_future.done():
            return None, None, None

        path   = result_future.result().result.path.poses
        elapsed = (t1-t0)*1000

        # Compute path length
        path_len = 0.0
        for i in range(1, len(path)):
            dx = (path[i].pose.position.x -
                  path[i-1].pose.position.x)
            dy = (path[i].pose.position.y -
                  path[i-1].pose.position.y)
            path_len += math.sqrt(dx**2+dy**2)

        return elapsed, path_len, len(path)

    # ─────────────────────────────────────────────
    # Main benchmark
    # ─────────────────────────────────────────────
    def run(self):
        verts = self.verts
        info  = self.occupancy_grid.info
        total_cells = info.width * info.height

        print('\n' + '='*75)
        print('  PLANNER BENCHMARK: Hybrid Topological vs Grid A* (Nav2)')
        print('='*75)
        print(f'  Map resolution : {info.resolution} m/cell')
        print(f'  Map size       : {info.width} x {info.height}'
              f' = {total_cells:,} cells')
        print(f'  Graph size     : {len(verts)} vertices, '
              f'{len(self.edges)} edges')
        print(f'  Search space   : grid is '
              f'{total_cells/len(verts):,.0f}x larger than graph')
        print('='*75)

        # Generate test pairs — pick vertices far apart
        import random
        random.seed(42)
        pairs = []
        ids   = list(range(len(verts)))
        attempts = 0
        while len(pairs) < 5 and attempts < 100:
            attempts += 1
            a, b = random.sample(ids, 2)
            d = math.sqrt(
                (verts[a]['x']-verts[b]['x'])**2 +
                (verts[a]['y']-verts[b]['y'])**2)
            if d > 4.0:
                pairs.append((a, b, d))

        print(f'\n{"#":<3} {"Pair":<12} '
              f'{"Topo(ms)":>10} {"Topo(m)":>9} {"Hops":>5} '
              f'{"A*(ms)":>10} {"A*(m)":>9} {"Poses":>7} '
              f'{"Speedup":>9}')
        print('-'*75)

        results     = []
        total_topo  = 0.0
        total_astar = 0.0
        valid       = 0

        for i, (a, b, straight_d) in enumerate(pairs):
            sx, sy = verts[a]['x'], verts[a]['y']
            gx, gy = verts[b]['x'], verts[b]['y']

            self.get_logger().info(
                f'Testing pair {i+1}/5: '
                f'v{a}({sx:.1f},{sy:.1f}) → '
                f'v{b}({gx:.1f},{gy:.1f})')

            topo_ms, topo_dist, topo_hops = self.plan_topo(
                sx, sy, gx, gy)
            astar_ms, astar_dist, astar_poses = self.plan_nav2(
                sx, sy, gx, gy)

            if astar_ms is None:
                print(f'{i+1:<3} v{a}→v{b:<8}  '
                      f'{topo_ms:>10.4f} {topo_dist:>9.2f} '
                      f'{topo_hops:>5}  '
                      f'{"FAILED":>10} {"---":>9} {"---":>7} '
                      f'{"---":>9}')
                continue

            speedup     = astar_ms / topo_ms
            total_topo  += topo_ms
            total_astar += astar_ms
            valid       += 1

            dist_ratio = (topo_dist/astar_dist*100
                          if astar_dist else 0)

            print(f'{i+1:<3} v{a}→v{b:<8}  '
                  f'{topo_ms:>10.4f} {topo_dist:>9.2f} '
                  f'{topo_hops:>5}  '
                  f'{astar_ms:>10.2f} {astar_dist:>9.2f} '
                  f'{astar_poses:>7}  '
                  f'{speedup:>8.1f}x')

            results.append({
                'pair'         : f'v{a}→v{b}',
                'straight_dist': round(straight_d, 2),
                'topo_ms'      : round(topo_ms, 4),
                'topo_dist_m'  : round(topo_dist, 2),
                'topo_hops'    : topo_hops,
                'astar_ms'     : round(astar_ms, 2),
                'astar_dist_m' : round(astar_dist, 2),
                'astar_poses'  : astar_poses,
                'speedup_x'    : round(speedup, 1),
                'path_len_ratio': round(dist_ratio, 1),
            })

        # Summary
        print('='*75)
        if valid > 0:
            avg_spd = total_astar / total_topo
            print(f'\n  SUMMARY ({valid} pairs):')
            print(f'  Topo total time : {total_topo:.4f} ms')
            print(f'  A*   total time : {total_astar:.2f} ms')
            print(f'  Average speedup : {avg_spd:.1f}x faster')
            print(f'\n  Search space reduction:')
            print(f'  Grid cells      : {total_cells:,}')
            print(f'  Topo vertices   : {len(verts)}')
            print(f'  Reduction       : {total_cells/len(verts):,.0f}x')
            print(f'\n  Path quality:')
            if results:
                avg_ratio = sum(
                    r['path_len_ratio'] for r in results
                    if r['path_len_ratio']) / len(results)
                print(f'  Topo path length: ~{avg_ratio:.1f}% of A* length')
                print(f'  (100% = same length, >100% = slightly longer)')
        print('='*75)

        # Save results
        out = os.path.expanduser(
            '~/gen_ws/src/amcc_nav/maps/benchmark_results.json')
        with open(out, 'w') as f:
            json.dump({
                'map_info': {
                    'width'     : info.width,
                    'height'    : info.height,
                    'cells'     : total_cells,
                    'resolution': info.resolution,
                },
                'graph_info': {
                    'vertices': len(verts),
                    'edges'   : len(self.edges),
                },
                'summary': {
                    'avg_speedup_x'       : round(avg_spd, 1)
                                            if valid > 0 else 0,
                    'search_space_ratio'  : round(
                        total_cells/len(verts), 1),
                },
                'pairs': results
            }, f, indent=2)
        print(f'\n  Results saved: benchmark_results.json')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PlannerBenchmark()
    node.timer_start = node.create_timer(1.0, node._start)
    rclpy.spin(node)

if __name__ == '__main__':
    main()