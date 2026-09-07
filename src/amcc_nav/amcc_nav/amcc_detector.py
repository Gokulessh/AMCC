#!/usr/bin/env python3
"""
amcc_landmark_detector.py
ROS2 node: subscribes to /scan, publishes InvAMCC value + landmarks
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry

import numpy as np
from collections import deque
import math


class AMCCDetector(Node):
    def __init__(self):
        super().__init__('amcc_detector')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('search_radius', 3.0)       # r in paper
        self.declare_parameter('inv_amcc_threshold', 0.4)  # Inv_th
        self.declare_parameter('delta_in', 1.0)            # inner dead zone
        self.declare_parameter('delta_out', 2.5)           # outer dead zone
        self.declare_parameter('max_scan_range', 10.0)     # filter far points

        self.r       = self.get_parameter('search_radius').value
        self.inv_th  = self.get_parameter('inv_amcc_threshold').value
        self.d_in    = self.get_parameter('delta_in').value
        self.d_out   = self.get_parameter('delta_out').value
        self.max_r   = self.get_parameter('max_scan_range').value

        # ── Subscribers ──────────────────────────────────────────────
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # ── Publishers ───────────────────────────────────────────────
        self.invamcc_pub    = self.create_publisher(Float32, '/inv_amcc', 10)
        self.landmark_pub   = self.create_publisher(MarkerArray, '/landmarks', 10)
        self.trajectory_pub = self.create_publisher(Marker, '/inv_amcc_trajectory', 10)

        # ── State ─────────────────────────────────────────────────────
        self.current_pose   = None          # (x, y)
        self.trajectory     = []            # list of (x, y, inv_amcc)
        self.vertices       = []            # selected landmark vertices
        self.marker_id      = 0

        self.get_logger().info('AMCC Detector node started')

    # ─────────────────────────────────────────────────────────────────
    # Odometry callback — track robot position
    # ─────────────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_pose = (x, y)

    # ─────────────────────────────────────────────────────────────────
    # Main scan callback
    # ─────────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        if self.current_pose is None:
            return

        # 1. Convert scan to Cartesian point cloud
        points = self._scan_to_cartesian(msg)
        if len(points) < 10:
            return

        # 2. Compute AMCC for this scan frame
        amcc_val, amcc_angle = self._compute_amcc(points)
        inv_amcc = 1.0 - amcc_val

        # 3. Store in trajectory
        px, py = self.current_pose
        self.trajectory.append((px, py, inv_amcc))

        # 4. Publish raw InvAMCC value
        msg_out = Float32()
        msg_out.data = inv_amcc
        self.invamcc_pub.publish(msg_out)

        # 5. Run vertex selection
        self._vertex_selection(px, py)

        # 6. Visualize
        self._publish_trajectory_marker()

    # ─────────────────────────────────────────────────────────────────
    # Step A: Polar → Cartesian, filter invalid/far readings
    # ─────────────────────────────────────────────────────────────────
    def _scan_to_cartesian(self, scan_msg):
        points = []
        angle = scan_msg.angle_min
        for r in scan_msg.ranges:
            if (scan_msg.range_min < r < min(scan_msg.range_max, self.max_r)
                    and not math.isnan(r) and not math.isinf(r)):
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append((x, y))
            angle += scan_msg.angle_increment
        return np.array(points)  # shape (N, 2)

    # ─────────────────────────────────────────────────────────────────
    # Step B: Core AMCC computation
    # ─────────────────────────────────────────────────────────────────
    def _compute_amcc(self, points):
        """
        Given Nx2 array of (x,y) scan points in robot frame,
        returns (amcc_value, amcc_angle).
        """
        X = points[:, 0]
        Y = points[:, 1]

        # Basic statistics
        var_x  = np.var(X)
        var_y  = np.var(Y)
        cov_xy = np.cov(X, Y, bias=True)[0, 1]  # σ_XY

        # ── Compute c_max (AMC value) ────────────────────────────────
        # c_max = sqrt( ((σ²_X - σ²_Y)/2)² + σ²_XY )
        diff_term = (var_x - var_y) / 2.0
        c_max = math.sqrt(diff_term**2 + cov_xy**2)

        # ── Compute AMCC = 2*c_max / (σ²_X + σ²_Y) ──────────────────
        denom = var_x + var_y
        if denom < 1e-9:
            return 0.0, 0.0

        amcc_val = 2.0 * c_max / denom
        amcc_val = min(amcc_val, 1.0)  # clamp numerical errors

        # ── Compute AMCC angle ────────────────────────────────────────
        # β = arcsin(σ_XY / c_max)
        if c_max < 1e-9:
            amcc_angle = 0.0
        else:
            beta = math.asin(max(-1.0, min(1.0, cov_xy / c_max)))
            # θ_AMCC = - sgn(σ_XY)·sgn(σ²_X - σ²_Y)·(π/4 - |β|/2)
            sign_cov  = np.sign(cov_xy)  if abs(cov_xy)        > 1e-9 else 0
            sign_diff = np.sign(var_x - var_y) if abs(var_x - var_y) > 1e-9 else 0
            amcc_angle = -sign_cov * sign_diff * (math.pi/4 - abs(beta)/2)

        return float(amcc_val), float(amcc_angle)

    # ─────────────────────────────────────────────────────────────────
    # Step C: Vertex selection — Algorithm 1 from paper
    # ─────────────────────────────────────────────────────────────────
    def _vertex_selection(self, cx, cy):
        """
        Implements the three-zone matching set segmentation:
          S_ew  (exploration waiting): dist < delta_in   → too close, wait
          S_rf  (redundancy filter):  dist > delta_out   → too far, skip
          S_mm  (maximum matching):   in between         → valid zone
        """
        if len(self.trajectory) < 5:
            return

        traj_arr = np.array([(t[0], t[1]) for t in self.trajectory])
        inv_arr  = np.array([t[2] for t in self.trajectory])

        # Find neighbors within search radius r
        dists = np.sqrt((traj_arr[:, 0] - cx)**2 + (traj_arr[:, 1] - cy)**2)
        neighbor_mask = dists < self.r
        neighbor_idx  = np.where(neighbor_mask)[0]

        if len(neighbor_idx) == 0:
            return

        neighbor_dists    = dists[neighbor_idx]
        neighbor_invamcc  = inv_arr[neighbor_idx]

        # Find the point with maximum InvAMCC among neighbors
        best_local = np.argmax(neighbor_invamcc)
        best_inv   = neighbor_invamcc[best_local]
        best_dist  = neighbor_dists[best_local]
        best_global_idx = neighbor_idx[best_local]

        # ── Threshold check (Fig 3d in paper) ────────────────────────
        if best_inv < self.inv_th:
            return   # corridor region, no landmark here

        # ── Three-zone check ─────────────────────────────────────────
        if best_dist < self.d_in:
            return   # S_ew: too early, keep moving
        if best_dist > self.d_out:
            return   # S_rf: already passed best chance

        # ── S_mm: select this as a vertex ────────────────────────────
        vx = traj_arr[best_global_idx, 0]
        vy = traj_arr[best_global_idx, 1]

        # Check not too close to existing vertices (deduplication)
        for (ex, ey, _) in self.vertices:
            if math.sqrt((vx-ex)**2 + (vy-ey)**2) < self.r / 3.0:
                return  # duplicate

        # Classify landmark type based on InvAMCC value
        landmark_type = self._classify_landmark(best_inv)

        self.vertices.append((vx, vy, best_inv))
        self.get_logger().info(
            f'[LANDMARK] {landmark_type} at ({vx:.2f}, {vy:.2f})'
            f'  InvAMCC={best_inv:.3f}')
        self._publish_vertex_marker(vx, vy, best_inv, landmark_type)

    # ─────────────────────────────────────────────────────────────────
    # Heuristic classification from InvAMCC magnitude
    # ─────────────────────────────────────────────────────────────────
    def _classify_landmark(self, inv_amcc):
        """
        Higher InvAMCC → more scattered scan (intersection/T-junction)
        Lower InvAMCC (just above threshold) → entrance/door
        These thresholds are environment-dependent — tune to your space.
        """
        if inv_amcc > 0.75:
            return 'INTERSECTION'
        elif inv_amcc > 0.60:
            return 'T_JUNCTION'
        elif inv_amcc > 0.50:
            return 'TURN'
        else:
            return 'DOOR_OR_ENTRANCE'

    # ─────────────────────────────────────────────────────────────────
    # Visualization helpers
    # ─────────────────────────────────────────────────────────────────
    def _publish_vertex_marker(self, x, y, inv_amcc, ltype):
        markers = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = 'landmarks'
        m.id     = self.marker_id
        m.type   = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.1
        m.scale.x = m.scale.y = m.scale.z = 0.4

        # Color by type
        color_map = {
            'INTERSECTION':    (1.0, 0.0, 0.0),
            'T_JUNCTION':      (1.0, 0.5, 0.0),
            'TURN':            (1.0, 1.0, 0.0),
            'DOOR_OR_ENTRANCE':(0.0, 1.0, 0.0),
        }
        r, g, b = color_map.get(ltype, (1.0, 1.0, 1.0))
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.9

        markers.markers.append(m)
        self.landmark_pub.publish(markers)
        self.marker_id += 1

    def _publish_trajectory_marker(self):
        if len(self.trajectory) < 2:
            return
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = 'inv_amcc_traj'
        m.id     = 0
        m.type   = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r = 0.0
        m.color.g = 0.8
        m.color.b = 0.8
        m.color.a = 0.7

        from geometry_msgs.msg import Point
        for (x, y, _) in self.trajectory[-500:]:  # last 500 pts
            p = Point()
            p.x, p.y, p.z = x, y, 0.0
            m.points.append(p)

        self.trajectory_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = AMCCDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()