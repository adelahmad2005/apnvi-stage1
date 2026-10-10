from setuptools import find_packages, setup

package_name = 'apnvi_stage2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adela',
    maintainer_email='adelahmad2005@gmail.com',
    description='APNVI Stage 2: the user asks for something and a red box goes around it',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stage2_node = apnvi_stage2.stage2_node:main',
        ],
    },
)
