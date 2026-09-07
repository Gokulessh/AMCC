#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from collections import deque
import matplotlib
matplotlib.use('TkAgg')          # force stable backend
import matplotlib.pyplot as plt

class AMCCPlotter(Node):
    def __init__(self):
        super().__init__('amcc_plotter')
        self.values    = deque(maxlen=200)
        self.threshold = 0.4
        self.sub = self.create_subscription(
            Float32, '/inv_amcc', self.callback, 10)
        self.get_logger().info('AMCC Plotter ready')

    def callback(self, msg):
        self.values.append(msg.data)

    def identify_structure(self, val):
        if val > 0.85:   return 'INTERSECTION'
        elif val > 0.70: return 'T-JUNCTION'
        elif val > 0.55: return 'TURN/CORNER'
        elif val > 0.40: return 'DOOR/ENTRANCE'
        else:            return 'CORRIDOR'


def main(args=None):
    rclpy.init(args=args)
    node = AMCCPlotter()

    # Setup plot ONCE in main thread
    plt.ion()                              # interactive mode — never blocks
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1e1e1e')
    ax.spines['right'].set_color('#1e1e1e')
    ax.set_xlabel('Samples', color='white')
    ax.set_ylabel('InvAMCC', color='white')

    # Static elements
    ax.axhline(y=0.4,  color='yellow', linestyle='--', alpha=0.7, label='Threshold')
    ax.axhspan(0.40, 0.55, alpha=0.15, color='green',  label='Door/Entrance')
    ax.axhspan(0.55, 0.70, alpha=0.15, color='orange', label='Turn')
    ax.axhspan(0.70, 0.85, alpha=0.15, color='red',    label='T-Junction')
    ax.axhspan(0.85, 1.10, alpha=0.15, color='purple', label='Intersection')
    ax.legend(loc='upper right', facecolor='#333333',
              labelcolor='white', fontsize=8)

    line, = ax.plot([], [], color='cyan', linewidth=1.5)
    fig.tight_layout()

    # Main loop — spin ROS + update plot together
    try:
        while rclpy.ok():
            # Process ROS callbacks (non-blocking, 50ms timeout)
            rclpy.spin_once(node, timeout_sec=0.05)

            # Update plot data
            if len(node.values) > 0:
                y = list(node.values)
                x = list(range(len(y)))
                line.set_data(x, y)
                ax.set_xlim(0, max(len(y), 50))

                current = y[-1]
                # Line color by zone
                if current > 0.85:   line.set_color('purple')
                elif current > 0.70: line.set_color('red')
                elif current > 0.55: line.set_color('orange')
                elif current > 0.40: line.set_color('green')
                else:                line.set_color('cyan')

                structure = node.identify_structure(current)
                ax.set_title(
                    f'InvAMCC | Current: {current:.3f} | {structure}',
                    color='white', fontsize=12)

                fig.canvas.draw()
                fig.canvas.flush_events()   # non-blocking GUI update

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close()


if __name__ == '__main__':
    main()