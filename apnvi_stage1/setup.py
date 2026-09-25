from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'apnvi_stage1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adela',
    maintainer_email='adelahmad2005@gmail.com',
    description='APNVI Stage 1: obstacle warning from the depth camera and ultrasonic',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hazard_node = apnvi_stage1.hazard_node:main',
            'output_node = apnvi_stage1.output_node:main',
            'esp32_reader = apnvi_stage1.esp32_reader:main',
        ],
    },
)
