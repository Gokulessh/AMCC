#!/usr/bin/env python3
"""
Loads hybrid_map_clean.json and publishes it to /hybrid_map_clean
for verification in RViz2.
"""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json,os

class GraphVisualizer(Node):
    def __init__(self):
        super().__init__('graph_visualizer')
        self.pub = self.create_publisher(
            MarkerArray, '/hybrid_map_clean', 10)
        self.create_timer(1.0, self.publish)

        with open(os.path.expanduser('~/gen_ws/src/amcc_nav/maps/hybrid_map_clean.json')) as f:
            data = json.load(f)
        self.verts = data['vertices']
        self.edges = data['edges']
        self.get_logger().info(
            f'Loaded: {len(self.verts)} vertices, '
            f'{len(self.edges)} edges')

    def publish(self):
        markers = MarkerArray()

        for v in self.verts:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns, m.id        = 'clean_v', v['id']
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position.x = v['x']
            m.pose.position.y = v['y']
            m.pose.position.z = 0.3
            m.scale.x = m.scale.y = m.scale.z = 0.6
            m.color.r, m.color.g = 0.0, 1.0
            m.color.b, m.color.a = 0.0, 1.0
            markers.markers.append(m)

            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp    = self.get_clock().now().to_msg()
            t.ns, t.id        = 'clean_label', v['id'] + 1000
            t.type            = Marker.TEXT_VIEW_FACING
            t.action          = Marker.ADD
            t.pose.position.x = v['x']
            t.pose.position.y = v['y']
            t.pose.position.z = 0.8
            t.scale.z         = 0.4
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text            = str(v['id'])
            markers.markers.append(t)

        for i, e in enumerate(self.edges):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns, m.id        = 'clean_e', i
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.1
            m.color.r, m.color.g = 0.0, 1.0
            m.color.b, m.color.a = 0.5, 1.0

            for vid in [e['u'], e['v']]:
                vv = self.verts[vid]
                p  = Point()
                p.x, p.y, p.z = vv['x'], vv['y'], 0.3
                m.points.append(p)
            markers.markers.append(m)

        self.pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GraphVisualizer())

if __name__ == '__main__':
    main()