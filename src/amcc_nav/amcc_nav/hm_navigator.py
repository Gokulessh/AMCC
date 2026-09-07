#!/usr/bin/env python3
"""
Listens to /goal_pose (from RViz2 2D Goal Pose button)
and navigates using the topological hybrid map.
No need to edit file — just click 2D Goal Pose in RViz2!
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import FollowWaypoints, NavigateToPose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import json, math, heapq, os

class HMNavigator(Node):
    def __init__(self):
        super().__init__('hm_navigator')

        self._waypoint_client = ActionClient(
            self, FollowWaypoints, 'follow_waypoints')
        self._nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        self.robot_x = 0.0
        self.robot_y = 0.0

        # Subscribe to odom for robot position
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Subscribe to RViz2 2D Goal Pose button
        self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)

        # Load cleaned graph
        try:
            with open(os.path.expanduser('~/gen_ws/src/amcc_nav/maps/hybrid_map_clean.json')) as f:
                data = json.load(f)
            self.verts = data['vertices']
            self.edges = data['edges']
            self.get_logger().info(
                f'Graph loaded: {len(self.verts)} vertices, '
                f'{len(self.edges)} edges')
            self.get_logger().info(
                'Ready! Click 2D Goal Pose in RViz2 to navigate.')
        except FileNotFoundError:
            self.get_logger().error(
                'hybrid_map_clean.json not found! '
                'Run graph_pruner first.')
            raise

    # ─────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def goal_callback(self, msg):
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        self.get_logger().info(
            f'New goal received: ({goal_x:.2f}, {goal_y:.2f})')
        self.navigate(goal_x, goal_y)

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────
    def nearest_vertex(self, x, y):
        best, best_d = 0, float('inf')
        for v in self.verts:
            d = math.sqrt((v['x']-x)**2 + (v['y']-y)**2)
            if d < best_d:
                best_d = d
                best   = v['id']
        return best

    def dijkstra(self, start_id, goal_id):
        n   = len(self.verts)
        INF = float('inf')
        dist = [INF] * n
        prev = [-1]  * n
        dist[start_id] = 0.0

        adj = {i: [] for i in range(n)}
        for e in self.edges:
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

        path, cur = [], goal_id
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        if not path or path[0] != start_id:
            return []
        return path

    # ─────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────
    def navigate(self, goal_x, goal_y):
        start_id = self.nearest_vertex(self.robot_x, self.robot_y)
        goal_id  = self.nearest_vertex(goal_x, goal_y)

        self.get_logger().info(
            f'Robot  → v{start_id} '
            f'({self.verts[start_id]["x"]:.2f}, '
            f'{self.verts[start_id]["y"]:.2f})')
        self.get_logger().info(
            f'Goal   → v{goal_id} '
            f'({self.verts[goal_id]["x"]:.2f}, '
            f'{self.verts[goal_id]["y"]:.2f})')

        path_ids = self.dijkstra(start_id, goal_id)

        if not path_ids:
            self.get_logger().error('No path found!')
            return

        self.get_logger().info(
            f'Path: {" → ".join(f"v{i}" for i in path_ids)}')

        # Build PoseStamped waypoints
        waypoints = []
        for vid in path_ids:
            v    = self.verts[vid]
            pose = PoseStamped()
            pose.header.frame_id    = 'map'
            pose.header.stamp       = self.get_clock().now().to_msg()
            pose.pose.position.x    = v['x']
            pose.pose.position.y    = v['y']
            pose.pose.position.z    = 0.0
            pose.pose.orientation.w = 1.0
            waypoints.append(pose)

        # Add exact clicked goal as final waypoint
        final = PoseStamped()
        final.header.frame_id    = 'map'
        final.header.stamp       = self.get_clock().now().to_msg()
        final.pose.position.x    = goal_x
        final.pose.position.y    = goal_y
        final.pose.orientation.w = 1.0
        waypoints.append(final)

        self.get_logger().info(
            f'Sending {len(waypoints)} waypoints to Nav2...')
        self._send_waypoints(waypoints)

    def _send_waypoints(self, waypoints):
        self.get_logger().info(
            'Waiting for FollowWaypoints action server...')
        self._waypoint_client.wait_for_server()

        goal_msg        = FollowWaypoints.Goal()
        goal_msg.poses  = waypoints

        future = self._waypoint_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal REJECTED by Nav2!')
            return
        self.get_logger().info('Goal ACCEPTED — robot moving!')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        missed = result.missed_waypoints
        if not missed:
            self.get_logger().info('✅ Navigation complete!')
        else:
            self.get_logger().warn(
                f'Done. Missed waypoints: {missed}')
        self.get_logger().info(
            'Ready for next goal — click 2D Goal Pose in RViz2')

    def feedback_callback(self, feedback):
        wp = feedback.feedback.current_waypoint
        self.get_logger().info(
            f'  Moving to waypoint {wp}...')


def main(args=None):
    rclpy.init(args=args)
    node = HMNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()