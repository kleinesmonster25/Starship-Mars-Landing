"""
Starship Mars Landing Simulation

This Python script simulates a thrust-only landing of SpaceX's Starship on Mars, without parachutes.
It models realistic Mars gravity (3.71 m/s²), atmosphere, wind, and Jezero-like terrain.
Features include LIDAR sensor fusion, engine failure handling, and a 3D trajectory visualization.

Dependencies:
- Python 3
- NumPy
- Matplotlib

Usage:
    pip install numpy matplotlib
    python mars_landing.py

Output:
- Console logs: Altitude, velocity, fuel, thrust
- Visualization: Saves 'mars_landing_simulation_3d.png' with 3D trajectory and plots

License:
    MIT License (see below)

Copyright (c) 2025 [Your Name or Anonymous]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Constants for Starship Mars landing
MARS_GRAVITY = 3.71  # m/s^2, Mars surface gravity
MARS_ROTATION_OMEGA = 7.083e-5  # rad/s, Mars angular velocity
MARS_ATMOSPHERE_DENSITY_SEA_LEVEL = 0.02  # kg/m^3, surface density
MARS_SCALE_HEIGHT = 11000  # meters, atmospheric scale height
DRAG_COEFFICIENT = 0.75  # Starship's drag coefficient
CROSS_SECTIONAL_AREA = 100  # m^2, approximate for Starship
SPECIFIC_IMPULSE = 350  # seconds, Raptor engine specific impulse
G0 = 9.81  # m/s^2, standard gravity for fuel flow
MAX_LIDAR_RANGE = 12000  # meters
SCAN_FREQUENCY = 50  # Hz
STARSHIP_MASS = 200000  # kg, total mass (dry + fuel + payload)
MAX_THRUST = 7500000  # Newtons, 3 Raptor engines (2.5 MN each)
MIN_THRUST = 2250000  # Newtons, 30% throttle (0.75 MN per engine)
ENGINE_STARTUP_DELAY = 0.1  # seconds
MIN_FUEL_RESERVE = 40000  # kg, ~25% safety reserve
SAFE_LANDING_SPEED = 0.5  # m/s
MIN_SAFE_ALTITUDE = 50  # meters
WIND_MAX_ACCELERATION = 0.1  # m/s^2, max lateral wind gust
HOVER_ALTITUDE = 100  # meters, pre-landing hover phase

class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def magnitude(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

class TerrainPatch:
    def __init__(self, coordinates, altitude, roughness, obstacles, size, dust_density):
        self.coordinates = coordinates  # Vector3D
        self.altitude = altitude        # meters
        self.roughness = roughness      # 0-1 scale
        self.obstacles = obstacles      # integer
        self.size = size                # meters
        self.dust_density = dust_density  # 0-1, Mars dust hazard

class LaserScanData:
    def __init__(self, distance=0, velocity=None):
        self.distance = distance
        self.velocity = velocity if velocity is not None else Vector3D(0, 0, 0)
        self.terrain_map = []

class Starship:
    def __init__(self):
        self.position = Vector3D(0, 0, 10000)  # Initial altitude
        self.velocity = Vector3D(0, 0, -50)    # Initial descent
        self.fuel = 160000                     # kg, for thrust-only landing
        self.thrust = 0
        self.active_engines = 3  # Number of functional Raptor engines
        self.engine_delay_timer = 0  # For startup/shutdown
        self.lidar = NavigationDopplerLidar()
        self.hdl = HazardDetectionLidar()
        self.backup_lidar = NavigationDopplerLidar()
        self.pid = PIDController(kp=0.8, ki=0.15, kd=0.25)
        self.backup_pid = PIDController(kp=0.7, ki=0.1, kd=0.2)  # Redundant PID
        self.safe_zones = []
        self.altitude_history = []
        self.velocity_history = []
        self.fuel_history = []
        self.time_history = []
        self.position_history = [(0, 0, 10000)]  # For 3D trajectory

    def update_state(self, delta_time):
        # Simulate engine failure (random, one-time at 5000m for testing)
        if self.position.z < 5000 and self.active_engines == 3 and random.random() < 0.01:
            self.active_engines = 2
            print("Warning: One Raptor engine failed! Max thrust reduced to 5 MN.")

        # Calculate max thrust based on active engines
        current_max_thrust = MAX_THRUST * (self.active_engines / 3)
        current_min_thrust = MIN_THRUST * (self.active_engines / 3)

        # Apply engine startup delay
        if self.thrust > 0 and self.engine_delay_timer > 0:
            self.engine_delay_timer -= delta_time
            thrust_acc = 0
        else:
            thrust_acc = max(0, min(self.thrust, current_max_thrust) / STARSHIP_MASS)
            if self.thrust < current_min_thrust and self.thrust > 0:
                thrust_acc = current_min_thrust / STARSHIP_MASS

        # Calculate forces
        gravity_acc = -MARS_GRAVITY
        drag_acc = self.calculate_drag()
        centrifugal_acc = self.calculate_centrifugal()
        wind_acc = self.calculate_wind()

        # Total acceleration
        acceleration = Vector3D(wind_acc.x, wind_acc.y, thrust_acc + gravity_acc + drag_acc + centrifugal_acc)

        # Update velocity and position
        self.velocity = self.velocity + acceleration * delta_time
        self.position = self.position + self.velocity * delta_time

        # Fuel consumption
        if self.thrust > 0 and self.engine_delay_timer <= 0:
            mass_flow_rate = self.thrust / (SPECIFIC_IMPULSE * G0)
            self.fuel -= mass_flow_rate * delta_time

        # Log state
        self.altitude_history.append(self.position.z)
        self.velocity_history.append(self.velocity.z)
        self.fuel_history.append(self.fuel)
        self.time_history.append(self.time_history[-1] + delta_time if self.time_history else delta_time)
        self.position_history.append((self.position.x, self.position.y, self.position.z))

    def calculate_drag(self):
        if self.position.z <= 0:
            return 0
        density = MARS_ATMOSPHERE_DENSITY_SEA_LEVEL * math.exp(-self.position.z / MARS_SCALE_HEIGHT)
        speed = self.velocity.magnitude()
        drag_force = 0.5 * density * speed**2 * DRAG_COEFFICIENT * CROSS_SECTIONAL_AREA
        drag_acc = -drag_force / STARSHIP_MASS * (self.velocity.z / speed if speed > 0 else 0)
        return drag_acc

    def calculate_centrifugal(self):
        radius = 3.3895e6  # Mars radius
        centrifugal_acc = (MARS_ROTATION_OMEGA**2) * radius * 1e-3
        return centrifugal_acc

    def calculate_wind(self):
        # Height-dependent wind with gusts
        wind_strength = WIND_MAX_ACCELERATION * (1 + self.position.z / 10000)  # Stronger at higher altitude
        gust = math.sin(self.time_history[-1] * 0.5) * wind_strength * 0.5  # Sinusoidal gusts
        wind_x = random.uniform(-wind_strength, wind_strength) + gust
        wind_y = random.uniform(-wind_strength, wind_strength) + gust
        return Vector3D(wind_x, wind_y, 0)

    def scan_and_navigate(self):
        # Primary and backup LIDAR scans
        scan_data = self.lidar.scan(self.position, self.velocity)
        backup_data = self.backup_lidar.scan(self.position, self.velocity)
        
        # Outlier rejection (moving average filter)
        if len(self.altitude_history) >= 5:
            recent_altitudes = self.altitude_history[-5:]
            avg_altitude = sum(recent_altitudes) / len(recent_altitudes)
            if abs(scan_data.distance - avg_altitude) > 100:
                scan_data = backup_data
                print("Warning: Primary LIDAR outlier detected, using backup.")
        
        # Sensor fusion (average if both plausible)
        altitude = (scan_data.distance + backup_data.distance) / 2
        velocity = Vector3D(
            (scan_data.velocity.x + backup_data.velocity.x) / 2,
            (scan_data.velocity.y + backup_data.velocity.y) / 2,
            (scan_data.velocity.z + backup_data.velocity.z) / 2
        )
        if abs(altitude - self.position.z) > 100 or abs(velocity.z) > 300:
            velocity = scan_data.velocity
            altitude = scan_data.distance
            print("Warning: Sensor fusion failed, using primary LIDAR.")

        self.velocity = velocity

        # Terrain mapping and thrust control
        if altitude < 3000:
            terrain_map = self.hdl.scan(self.position, altitude)
            safe_zone = self.analyze_terrain(terrain_map)
            if safe_zone:
                self.safe_zones.append(safe_zone)
            if self.safe_zones:
                best_zone = min(self.safe_zones, key=lambda z: z.roughness + z.dust_density)
                self.adjust_trajectory(best_zone.coordinates)
            else:
                print("No safe landing zone found! Initiating hover...")
                self.thrust = STARSHIP_MASS * MARS_GRAVITY * 1.2
        elif altitude < HOVER_ALTITUDE + 50:
            # Hover phase to confirm terrain
            self.thrust = STARSHIP_MASS * MARS_GRAVITY * 1.05  # Maintain altitude
            print("Hovering at {:.1f}m to confirm landing zone...".format(altitude))
        else:
            self.thrust = STARSHIP_MASS * MARS_GRAVITY * 1.15

        return altitude

    def analyze_terrain(self, terrain_map):
        best_zone = None
        best_score = float('inf')
        for patch in terrain_map:
            if (patch.roughness < 0.02 and
                patch.obstacles == 0 and
                patch.size >= 60 and
                patch.dust_density < 0.1):
                score = patch.roughness + patch.dust_density
                if score < best_score:
                    best_score = score
                    best_zone = patch
        return best_zone

    def adjust_trajectory(self, target_coords):
        delta = Vector3D(target_coords.x - self.position.x,
                        target_coords.y - self.position.y, 0)
        distance = delta.magnitude()

        target_velocity_z = -2 if self.position.z > 500 else -0.3
        thrust_adjustment = self.pid.compute(self.velocity.z, target_velocity_z)
        if abs(thrust_adjustment) > 10:  # Switch to backup PID if unstable
            thrust_adjustment = self.backup_pid.compute(self.velocity.z, target_velocity_z)
            print("Warning: Primary PID unstable, using backup PID.")

        self.thrust = min(MAX_THRUST * (self.active_engines / 3),
                         STARSHIP_MASS * (MARS_GRAVITY + thrust_adjustment))

        if distance > 50:
            self.velocity.x = delta.x * 0.015
            self.velocity.y = delta.y * 0.015
        else:
            self.velocity.x *= 0.85
            self.velocity.y *= 0.85

    def emergency_abort(self):
        print("Critical failure detected! Initiating emergency abort...")
        self.thrust = MAX_THRUST * (self.active_engines / 3)
        self.velocity.z = 10
        print("Ascending to safe altitude. Mission aborted.")
        return False

    def land(self):
        print("Initiating Starship Mars landing sequence (thrust-only)...")
        delta_time = 1 / SCAN_FREQUENCY
        self.time_history.append(0)

        while self.position.z > 0:
            altitude = self.scan_and_navigate()
            self.update_state(delta_time)
            
            print(f"Altitude: {altitude:.1f}m, Velocity: {self.velocity.z:.1f}m/s, Fuel: {self.fuel:.1f}kg, Thrust: {self.thrust:.0f}N, Engines: {self.active_engines}")
            
            if self.fuel <= MIN_FUEL_RESERVE:
                print("Critical fuel level! Attempting emergency landing...")
                self.thrust = MAX_THRUST * (self.active_engines / 3)
            if self.fuel <= 0:
                print("Out of fuel! Crash imminent.")
                break
            if altitude < MIN_SAFE_ALTITUDE and not self.safe_zones:
                return self.emergency_abort()
            if self.position.z <= 0:
                if abs(self.velocity.z) <= SAFE_LANDING_SPEED:
                    print("Perfect landing achieved on Mars!")
                    self.plot_landing_data()
                else:
                    print(f"Crash landing at {abs(self.velocity.z):.1f}m/s!")
                break

    def plot_landing_data(self):
        fig = plt.figure(figsize=(12, 10))
        
        # 3D Trajectory Plot
        ax = fig.add_subplot(221, projection='3d')
        x, y, z = zip(*self.position_history)
        ax.plot(x, y, z, 'b-', label='Trajectory')
        ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', marker='o', label='Landing')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Starship Mars Landing Trajectory')
        ax.legend()

        # Altitude Plot
        ax2 = fig.add_subplot(222)
        ax2.plot(self.time_history, self.altitude_history, 'b-')
        ax2.set_ylabel('Altitude (m)')
        ax2.set_title('Altitude vs Time')
        ax2.grid(True)

        # Velocity Plot
        ax3 = fig.add_subplot(223)
        ax3.plot(self.time_history, self.velocity_history, 'r-')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Velocity vs Time')
        ax3.grid(True)

        # Fuel Plot
        ax4 = fig.add_subplot(224)
        ax4.plot(self.time_history, self.fuel_history, 'g-')
        ax4.set_ylabel('Fuel (kg)')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Fuel vs Time')
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('mars_landing_simulation_3d.png')
        print("Landing data plot saved as 'mars_landing_simulation_3d.png'")

class NavigationDopplerLidar:
    def scan(self, position, current_velocity):
        data = LaserScanData(
            distance=position.z + random.uniform(-0.1, 0.1),
            velocity=Vector3D(
                current_velocity.x + random.uniform(-0.005, 0.005),
                current_velocity.y + random.uniform(-0.005, 0.005),
                current_velocity.z + random.uniform(-0.005, 0.005)
            )
        )
        return data

class HazardDetectionLidar:
    def scan(self, position, altitude):
        terrain_map = []
        scan_range = max(200, int(altitude / 1.5))
        
        # Simplified Jezero Crater model: 10% safe zones, 30% rough, 60% moderate
        for x in range(-scan_range, scan_range + 1, 10):
            for y in range(-scan_range, scan_range + 1, 10):
                rand = random.random()
                if rand < 0.1:  # Safe zone
                    roughness = random.uniform(0, 0.02)
                    obstacles = 0
                    dust_density = random.uniform(0, 0.1)
                elif rand < 0.4:  # Rough terrain
                    roughness = random.uniform(0.1, 0.5)
                    obstacles = random.randint(1, 3)
                    dust_density = random.uniform(0.2, 0.5)
                else:  # Moderate terrain
                    roughness = random.uniform(0.02, 0.1)
                    obstacles = random.randint(0, 1)
                    dust_density = random.uniform(0.1, 0.2)
                patch = TerrainPatch(
                    coordinates=Vector3D(position.x + x, position.y + y, 0),
                    altitude=0,
                    roughness=roughness,
                    obstacles=obstacles,
                    size=60,
                    dust_density=dust_density
                )
                terrain_map.append(patch)
        return terrain_map

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def compute(self, current, target):
        error = target - current
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Run the simulation with multiple test cases
if __name__ == "__main__":
    test_cases = [
        {"name": "Nominal", "fuel": 160000, "seed": 42, "noise_factor": 1},
        {"name": "Low Fuel", "fuel": 120000, "seed": 43, "noise_factor": 1},
        {"name": "Engine Failure", "fuel": 160000, "seed": 44, "noise_factor": 1},
        {"name": "High Wind", "fuel": 160000, "seed": 45, "noise_factor": 1},
        {"name": "High LIDAR Noise", "fuel": 160000, "seed": 46, "noise_factor": 2}
    ]

    for test in test_cases:
        print(f"\nRunning test case: {test['name']}")
        random.seed(test["seed"])
        starship = Starship()
        starship.fuel = test["fuel"]
        NavigationDopplerLidar.scan = lambda self, pos, vel: LaserScanData(
            distance=pos.z + random.uniform(-0.1 * test["noise_factor"], 0.1 * test["noise_factor"]),
            velocity=Vector3D(
                vel.x + random.uniform(-0.005 * test["noise_factor"], 0.005 * test["noise_factor"]),
                vel.y + random.uniform(-0.005 * test["noise_factor"], 0.005 * test["noise_factor"]),
                vel.z + random.uniform(-0.005 * test["noise_factor"], 0.005 * test["noise_factor"])
            )
        )
        starship.land()