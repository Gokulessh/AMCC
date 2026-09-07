from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'amcc_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gokul',
    maintainer_email='127179014@satra.ac.in',
    description='AMCC landmark detection for hybrid mapping',
    license='MIT',
    entry_points={
        'console_scripts': [
            'amcc_detector      = amcc_nav.amcc_detector:main',
            'hybrid_map_builder = amcc_nav.hybrid_map_builder:main',
            'amcc_plotter       = amcc_nav.amcc_plotter:main',
            'graph_pruner = amcc_nav.graph_pruner:main',
            'graph_visualizer = amcc_nav.graph_visualizer:main',
            'hm_path_test = amcc_nav.hm_path_test:main',
            'hm_navigator = amcc_nav.hm_navigator:main',
            'map_snap = amcc_nav.map_snapshot:main',
            'planner_benchmark = amcc_nav.planner_benchmark:main',
        ],
    },
)
