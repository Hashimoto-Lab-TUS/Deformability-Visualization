from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'my_dynamixel_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ('share/ament_index/resource_index/packages', [os.path.join('resource', package_name)]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
        #('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        #('share/' + package_name, ['package.xml']),
        #('share/' + package_name + '/launch', glob('launch/*.py')),
        #('share/' + package_name + '/launch', ['launch/multi_node_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='reo',
    maintainer_email='4525562@ed.tus.ac.jp',
    description='Dynamixel controller package with service client',
    #license='MIT',
    license='Apache-2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'call_logger_client = my_dynamixel_controller.call_logger_client:main',
            'arduino_controller_node = my_dynamixel_controller.arduino_controller_node:main',
        ],
    },
)
