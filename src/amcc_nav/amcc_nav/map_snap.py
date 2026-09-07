#!/usr/bin/env python3
"""
Generates:
1. hybrid_map.png        — full map with all vertices and edges
2. landmark_X.png        — cropped image around each landmark
Reads hybrid_map_clean.json and the saved map .pgm/.yaml
"""
import json
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
import sys

# ─────────────────────────────────────────────────────────────
# CONFIGURE THESE PATHS
# ─────────────────────────────────────────────────────────────
MAP_YAML   = os.path.expanduser('/home/gokul/gen_ws/rc_map_save.yaml')
GRAPH_JSON = os.path.expanduser('~/gen_ws/src/amcc_nav/maps/hybrid_map_clean.json')
OUTPUT_DIR = os.path.expanduser('~/hybrid_map_images')
# ─────────────────────────────────────────────────────────────

def load_map(yaml_path):
    """Load map image and metadata from yaml."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    # Get image path relative to yaml
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(
            os.path.dirname(yaml_path), img_path)

    img      = Image.open(img_path).convert('L')
    img_arr  = np.array(img)

    res      = meta['resolution']          # metres per pixel
    origin   = meta['origin']              # [x, y, θ]

    print(f'Map loaded: {img_arr.shape[1]}x{img_arr.shape[0]} px')
    print(f'Resolution: {res} m/px')
    print(f'Origin    : {origin}')

    return img_arr, res, origin

def world_to_pixel(wx, wy, res, origin, img_height):
    """Convert world (metres) to image pixel coordinates."""
    px = int((wx - origin[0]) / res)
    py = int(img_height - (wy - origin[1]) / res)
    return px, py

def load_graph():
    with open(GRAPH_JSON) as f:
        return json.load(f)

def plot_full_hybrid_map(img_arr, res, origin, verts, edges):
    """Plot the full map with all vertices and edges."""
    h, w   = img_arr.shape
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    fig.patch.set_facecolor('#1a1a2e')

    # Show map
    ax.imshow(img_arr, cmap='gray', origin='upper',
              extent=[origin[0],
                      origin[0] + w * res,
                      origin[1],
                      origin[1] + h * res])

    ax.set_facecolor('#1a1a2e')
    ax.set_title('Hybrid Map — Topological Graph',
                 color='white', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X (metres)', color='white', fontsize=12)
    ax.set_ylabel('Y (metres)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    # Draw edges first (behind vertices)
    for e in edges:
        u = verts[e['u']]
        v = verts[e['v']]
        ax.plot([u['x'], v['x']],
                [u['y'], v['y']],
                color='#00aaff', linewidth=1.5,
                alpha=0.8, zorder=2)

        # Edge weight label at midpoint
        mid_x = (u['x'] + v['x']) / 2
        mid_y = (u['y'] + v['y']) / 2
        ax.text(mid_x, mid_y,
                f"{e['weight']:.1f}m",
                color='#aaddff', fontsize=6,
                ha='center', va='center', zorder=3,
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='#1a1a2e',
                          alpha=0.6, edgecolor='none'))

    # Draw vertices
    for v in verts:
        ax.scatter(v['x'], v['y'],
                   s=120, color='#ff4444',
                   edgecolors='white', linewidths=1.0,
                   zorder=4)
        ax.text(v['x'] + 0.15, v['y'] + 0.15,
                f"v{v['id']}",
                color='white', fontsize=8,
                fontweight='bold', zorder=5)

    # Legend
    v_patch = mpatches.Patch(color='#ff4444', label='Landmark vertex')
    e_patch = mpatches.Patch(color='#00aaff', label='Topological edge')
    ax.legend(handles=[v_patch, e_patch],
              loc='upper right',
              facecolor='#2a2a4a',
              labelcolor='white',
              fontsize=10)

    # Stats
    ax.text(0.02, 0.02,
            f'Vertices: {len(verts)}   Edges: {len(edges)}',
            transform=ax.transAxes,
            color='#aaaaaa', fontsize=10,
            verticalalignment='bottom')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'hybrid_map.png')
    plt.savefig(out, dpi=150,
                bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Saved: {out}')
    plt.close()

def plot_landmark_crops(img_arr, res, origin, verts, edges):
    """Save a cropped image around each landmark vertex."""
    h, w     = img_arr.shape
    crop_m   = 3.0   # metres around each landmark to crop

    for v in verts:
        vx, vy = v['x'], v['y']
        vid    = v['id']

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#1a1a2e')

        # Show map cropped around vertex
        x_min = vx - crop_m
        x_max = vx + crop_m
        y_min = vy - crop_m
        y_max = vy + crop_m

        ax.imshow(img_arr, cmap='gray', origin='upper',
                  extent=[origin[0],
                          origin[0] + w * res,
                          origin[1],
                          origin[1] + h * res])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_facecolor('#1a1a2e')

        # Draw connected edges
        for e in edges:
            if e['u'] == vid or e['v'] == vid:
                other_id = e['v'] if e['u'] == vid else e['u']
                other    = verts[other_id]
                ax.plot([vx, other['x']],
                        [vy, other['y']],
                        color='#00aaff',
                        linewidth=2.0, alpha=0.9, zorder=2)
                # Other vertex (smaller)
                ax.scatter(other['x'], other['y'],
                           s=60, color='#ff8888',
                           edgecolors='white',
                           linewidths=0.8, zorder=3)
                ax.text(other['x'] + 0.1, other['y'] + 0.1,
                        f"v{other_id}",
                        color='#ffcccc', fontsize=8, zorder=4)

        # Draw this vertex (highlighted)
        ax.scatter(vx, vy,
                   s=200, color='#ff0000',
                   edgecolors='white',
                   linewidths=1.5, zorder=5)
        ax.text(vx + 0.1, vy + 0.15,
                f"v{vid}",
                color='white', fontsize=10,
                fontweight='bold', zorder=6)

        # Count connections
        connections = sum(
            1 for e in edges
            if e['u'] == vid or e['v'] == vid)

        ax.set_title(
            f'Landmark v{vid} | '
            f'({vx:.2f}, {vy:.2f}) | '
            f'{connections} connections',
            color='white', fontsize=10, pad=8)
        ax.set_xlabel('X (m)', color='white', fontsize=8)
        ax.set_ylabel('Y (m)', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#444444')

        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f'landmark_{vid}.png')
        plt.savefig(out, dpi=120,
                    bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  Saved: landmark_{vid}.png')
        plt.close()

def plot_graph_only(verts, edges):
    """
    Plot just the topological graph without map background.
    Clean diagram showing graph structure.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.set_title('Topological Graph Structure',
                 color='white', fontsize=16,
                 fontweight='bold', pad=15)
    ax.set_xlabel('X (metres)', color='white', fontsize=12)
    ax.set_ylabel('Y (metres)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.grid(True, color='#222233', linewidth=0.5, alpha=0.5)

    # Edges
    for e in edges:
        u = verts[e['u']]
        v = verts[e['v']]
        ax.annotate('',
            xy=(v['x'], v['y']),
            xytext=(u['x'], u['y']),
            arrowprops=dict(
                arrowstyle='<->',
                color='#00aaff',
                lw=1.5))
        mid_x = (u['x'] + v['x']) / 2
        mid_y = (u['y'] + v['y']) / 2
        ax.text(mid_x, mid_y,
                f"{e['weight']:.1f}m",
                color='#66bbff', fontsize=7,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='#0d0d1a',
                          alpha=0.8,
                          edgecolor='#224466'))

    # Vertices
    for v in verts:
        ax.scatter(v['x'], v['y'],
                   s=150, color='#ff5555',
                   edgecolors='white',
                   linewidths=1.2, zorder=5)
        ax.annotate(f"  v{v['id']}",
                    (v['x'], v['y']),
                    color='white', fontsize=9,
                    fontweight='bold', zorder=6)

    ax.text(0.02, 0.98,
            f'V={len(verts)}  E={len(edges)}',
            transform=ax.transAxes,
            color='#888888', fontsize=10,
            va='top')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'topo_graph_only.png')
    plt.savefig(out, dpi=150,
                bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Saved: {out}')
    plt.close()


def main():
    # Setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Output directory: {OUTPUT_DIR}')

    # Load data
    print('\nLoading map...')
    try:
        img_arr, res, origin = load_map(MAP_YAML)
    except FileNotFoundError:
        print(f'ERROR: Map not found at {MAP_YAML}')
        print('Update MAP_YAML path at top of script')
        sys.exit(1)

    print('Loading graph...')
    try:
        data  = load_graph()
    except FileNotFoundError:
        print('ERROR: hybrid_map_clean.json not found')
        print('Run graph_pruner first!')
        sys.exit(1)

    verts = data['vertices']
    edges = data['edges']
    print(f'Graph: {len(verts)} vertices, {len(edges)} edges')

    # Generate images
    print('\n1. Generating full hybrid map image...')
    plot_full_hybrid_map(img_arr, res, origin, verts, edges)

    print('\n2. Generating individual landmark crops...')
    plot_landmark_crops(img_arr, res, origin, verts, edges)

    print('\n3. Generating topology-only graph diagram...')
    plot_graph_only(verts, edges)

    print(f'\nDone! All images saved to: {OUTPUT_DIR}')
    print('Files:')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f'  {f:30s}  {size/1024:.1f} KB')


if __name__ == '__main__':
    main()