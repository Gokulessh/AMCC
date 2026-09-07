#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
import math
import json, os
import numpy as np

class HybridMapBuilder(Node):
    def __init__(self):
        super().__init__('hybrid_map_builder')

        self.topo_vertices  = []
        self.topo_edges     = []
        self.last_vertex_id = None
        self.occupancy_grid = None
        self.inflated_map   = None

        self.max_edge_length    = 7.0
        self.min_vertex_dist    = 1.5
        self.wall_check_samples = 30
        self.inflation_radius   = 0.3

        # ── Load existing graph if it exists ────────────────────────
        self._load_existing_graph()

        self.create_subscription(
            MarkerArray,   '/landmarks', self.landmark_callback, 10)
        self.create_subscription(
            OccupancyGrid, '/map',       self.map_callback,      10)

        self.graph_pub = self.create_publisher(
            MarkerArray, '/hybrid_map', 10)

        self.create_timer(5.0, self.save_graph)
        self.get_logger().info('Hybrid Map Builder started')
        maps_path = os.path.expanduser(
            '~/gen_ws/src/amcc_nav/maps/hybrid_map.json')
        if os.path.exists(maps_path):
            self.get_logger().warn(
                '⚠️  hybrid_map.json already exists — '
                'will be overwritten as you drive!')
            self.get_logger().warn(
                'Ctrl+C now if this is unintentional!')
            import time
            time.sleep(5.0)   # 5 second window to cancel
    def _load_existing_graph(self):
        """Load existing graph so we can continue adding to it."""
        path = os.path.expanduser(
            '~/gen_ws/src/amcc_nav/maps/hybrid_map.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.topo_vertices = data['vertices']
            self.topo_edges    = data['edges']
            # Set last vertex id to continue from where we left off
            if self.topo_vertices:
                self.last_vertex_id = self.topo_vertices[-1]['id']
            self.get_logger().info(
                f'Loaded existing graph: '
                f'{len(self.topo_vertices)} vertices, '
                f'{len(self.topo_edges)} edges')
            self.get_logger().info(
                'Continuing from existing map — drive to unconnected areas!')
        else:
            self.get_logger().info('No existing graph — starting fresh')
    # ─────────────────────────────────────────────
    # Map callback
    # ─────────────────────────────────────────────
    def map_callback(self, msg):
        self.occupancy_grid = msg
        self._build_inflated_map(msg)
        self.get_logger().info('Map received and inflated')

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

    # ─────────────────────────────────────────────
    # Landmark callback
    # ─────────────────────────────────────────────
    def landmark_callback(self, msg):
        for marker in msg.markers:
            vx = marker.pose.position.x
            vy = marker.pose.position.y

            # ── Deduplicate ─────────────────────────────────────────
            too_close = False
            for v in self.topo_vertices:
                if math.sqrt((v['x']-vx)**2 + (v['y']-vy)**2) \
                        < self.min_vertex_dist:
                    too_close = True
                    break
            if too_close:
                return

            # ── Add vertex ──────────────────────────────────────────
            vid = len(self.topo_vertices)
            self.topo_vertices.append({
                'id': vid, 'x': vx, 'y': vy, 'type': 'landmark'
            })
            self.get_logger().info(
                f'Vertex {vid} added at ({vx:.2f}, {vy:.2f})')

            # ── Connect to previous vertex ──────────────────────────
            if self.last_vertex_id is not None:
                self._try_add_edge(self.last_vertex_id, vid)

            # ── Connect to all nearby older vertices ────────────────
            for v in self.topo_vertices[:-1]:
                if v['id'] == self.last_vertex_id:
                    continue
                dist = math.sqrt((v['x']-vx)**2 + (v['y']-vy)**2)
                if dist < self.max_edge_length:
                    already = any(
                        (e['u'] == v['id'] and e['v'] == vid) or
                        (e['u'] == vid     and e['v'] == v['id'])
                        for e in self.topo_edges
                    )
                    if not already:
                        self._try_add_edge(v['id'], vid)

            self.last_vertex_id = vid
            self._publish_graph()

    # ─────────────────────────────────────────────
    # Edge validation
    # ─────────────────────────────────────────────
    def _try_add_edge(self, uid, vid):
        ux = self.topo_vertices[uid]['x']
        uy = self.topo_vertices[uid]['y']
        vx = self.topo_vertices[vid]['x']
        vy = self.topo_vertices[vid]['y']
        dist = math.sqrt((ux-vx)**2 + (uy-vy)**2)

        if dist > self.max_edge_length:
            self.get_logger().warn(
                f'Edge {uid}→{vid} REJECTED: too long ({dist:.2f}m)')
            return

        if not self._is_path_free(ux, uy, vx, vy):
            self.get_logger().warn(
                f'Edge {uid}→{vid} REJECTED: crosses wall')
            return

        self.topo_edges.append({
            'u': uid, 'v': vid, 'weight': round(dist, 3)
        })
        self.get_logger().info(
            f'Edge {uid}→{vid} ADDED ({dist:.2f}m)')

    def _is_path_free(self, x1, y1, x2, y2):
        # ── If no map yet, ALLOW the edge ──────────────────────────
        # Will be validated later when map arrives
        if self.inflated_map is None or self.occupancy_grid is None:
            self.get_logger().warn(
                'No map yet — allowing edge (will recheck later)')
            return True          # ← KEY FIX: was False before

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

    # ─────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────
    def _publish_graph(self):
        markers = MarkerArray()

        for v in self.topo_vertices:
            m = Marker()
            m.header.frame_id    = 'map'
            m.header.stamp       = self.get_clock().now().to_msg()
            m.ns                 = 'topo_v'
            m.id                 = v['id']
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = v['x']
            m.pose.position.y    = v['y']
            m.pose.position.z    = 0.2
            m.scale.x = m.scale.y = m.scale.z = 0.5
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
            m.color.a = 1.0
            markers.markers.append(m)

        for v in self.topo_vertices:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = 'topo_label'
            m.id              = v['id'] + 1000
            m.type            = Marker.TEXT_VIEW_FACING
            m.action          = Marker.ADD
            m.pose.position.x = v['x']
            m.pose.position.y = v['y']
            m.pose.position.z = 0.7
            m.scale.z         = 0.35
            m.color.r = m.color.g = m.color.b = m.color.a = 1.0
            m.text            = str(v['id'])
            markers.markers.append(m)

        for i, e in enumerate(self.topo_edges):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = 'topo_e'
            m.id              = i
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.08
            m.color.r         = 0.2
            m.color.g         = 0.6
            m.color.b         = 1.0
            m.color.a         = 0.9

            for vid in [e['u'], e['v']]:
                vv = self.topo_vertices[vid]
                p  = Point()
                p.x, p.y, p.z = vv['x'], vv['y'], 0.2
                m.points.append(p)
            markers.markers.append(m)

        self.graph_pub.publish(markers)

    # ─────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────
    def save_graph(self):
        if not self.topo_vertices:
            return

        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        data = {'vertices': self.topo_vertices, 'edges': self.topo_edges}

        # Always save latest
        latest_path = os.path.expanduser(
            '~/gen_ws/src/amcc_nav/maps/hybrid_map.json')
        with open(latest_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Also save timestamped backup — never overwritten
        backup_path = os.path.expanduser(
            f'~/gen_ws/src/amcc_nav/maps/hybrid_map_{timestamp}.json')
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(
            f'Saved: {len(self.topo_vertices)} vertices, '
            f'{len(self.topo_edges)} edges')
        self.get_logger().info(f'Backup: hybrid_map_{timestamp}.json')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(HybridMapBuilder())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
