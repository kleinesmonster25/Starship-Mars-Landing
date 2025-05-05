import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Fix für MacOS plot saving
plt.switch_backend('Agg')

# Constants for Starship Mars landing
MARS_GRAVITY = 3.71  # m/s^2, Mars surface gravity
MARS_ROTATION_OMEGA = 7.083e-5  # rad/s, Mars angular velocity
MARS_ATMOSPHERE_DENSITY_SEA_LEVEL = 0.05  # kg/m^3, erhöht für realistischeren Drag
MARS_SCALE_HEIGHT = 11000  # meters, atmospheric scale height
DRAG_COEFFICIENT = 0.75  # Starship's drag coefficient
CROSS_SECTIONAL_AREA = 100  # m^2, approximate for Starship
SPECIFIC_IMPULSE = 350  # seconds, Raptor engine specific impulse
G0 = 9.81  # m/s^2, standard gravity for fuel flow
MAX_LIDAR_RANGE = 12000  # meters
SCAN_FREQUENCY = 100  # Hz (angepasst für kleinere Schritte)
STARSHIP_MASS = 200000  # kg, total mass (dry + fuel + payload)
MAX_THRUST = 7500000  # Newtons, 3 Raptor engines (2.5 MN each)
MIN_THRUST = 2250000  # Newtons, 30% throttle (0.75 MN per engine)
ENGINE_STARTUP_DELAY = 0.1  # seconds
MIN_FUEL_RESERVE = 40000  # kg, ~20% safety reserve bei 200.000 kg
SAFE_LANDING_SPEED = 0.5  # m/s
MIN_SAFE_ALTITUDE = 50  # meters
WIND_MAX_ACCELERATION = 0.05  # m/s^2, reduzierte Windstärke für Stabilität
HOVER_ALTITUDE = 100  # meters, pre-landing hover phase
MAX_ALTITUDE = 100000  # 100km als sinnvolle Obergrenze
MAX_VELOCITY = 300  # Reduziert für realistischere Simulation


class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)


class TerrainPatch:
    def __init__(self, coordinates, altitude, roughness, obstacles, size, dust_density):
        self.coordinates = coordinates  # Vector3D
        self.altitude = altitude  # meters
        self.roughness = roughness  # 0-1 scale
        self.obstacles = obstacles  # integer
        self.size = size  # meters
        self.dust_density = dust_density  # 0-1, Mars dust hazard


class LaserScanData:
    def __init__(self, distance=0, velocity=None):
        self.distance = distance
        self.velocity = velocity if velocity is not None else Vector3D(0, 0, 0)
        self.terrain_map = []


class Starship:
    def __init__(self):
        self.position = Vector3D(0, 0, 10000)  # Initial altitude
        self.velocity = Vector3D(0, 0, -50)  # Initial descent
        self.fuel = 200000  # Erhöht auf 200.000 kg
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
        self.velocity_smoothing = []  # Für gleitenden Durchschnitt

    def update_state(self, delta_time):
        if self.fuel <= 0:
            self.fuel = 0
            self.thrust = 0

        if self.position.z < 5000 and self.active_engines == 3 and random.random() < 0.01:
            self.active_engines = 2
            print("Warning: One Raptor engine failed! Max thrust reduced to 5 MN.")

        current_max_thrust = MAX_THRUST * (self.active_engines / 3)
        current_min_thrust = MIN_THRUST * (self.active_engines / 3)

        if self.thrust > 0 and self.engine_delay_timer > 0:
            self.engine_delay_timer -= delta_time
            thrust_acc = 0
        else:
            thrust_acc = max(0, min(self.thrust, current_max_thrust) / STARSHIP_MASS)
            if self.thrust < current_min_thrust and self.thrust > 0:
                thrust_acc = current_min_thrust / STARSHIP_MASS

        # Dynamische Schubanpassung mit PID-Steuerung
        target_velocity_z = -10  # Langsamer, kontrollierter Abstieg
        thrust_adjustment = self.pid.compute(self.velocity.z, target_velocity_z)
        if math.isnan(thrust_adjustment) or math.isinf(thrust_adjustment) or abs(thrust_adjustment) > 10:
            thrust_adjustment = self.backup_pid.compute(self.velocity.z, target_velocity_z)
            print("Warning: Primary PID unstable, using backup PID.")
            if math.isnan(thrust_adjustment) or math.isinf(thrust_adjustment):
                thrust_adjustment = 0

        self.thrust = min(STARSHIP_MASS * (MARS_GRAVITY + thrust_adjustment), current_max_thrust)

        # Treibstoffschutz
        if self.fuel < 20000:  # 10% Reserve bei 200.000 kg
            self.thrust = min(self.thrust, STARSHIP_MASS * MARS_GRAVITY * 0.5)

        gravity_acc = -MARS_GRAVITY
        drag_acc = self.calculate_drag()
        centrifugal_acc = self.calculate_centrifugal()
        wind_acc = self.calculate_wind()

        acceleration = Vector3D(wind_acc.x, wind_acc.y, thrust_acc + gravity_acc + drag_acc + centrifugal_acc)
        self.velocity = self.velocity + acceleration * delta_time
        self.position = self.position + self.velocity * delta_time

        if self.thrust > 0 and self.engine_delay_timer <= 0:
            mass_flow_rate = self.thrust / (SPECIFIC_IMPULSE * G0)
            fuel_used = mass_flow_rate * delta_time
            if fuel_used > self.fuel:
                fuel_used = self.fuel
            self.fuel -= fuel_used

        # Geschwindigkeit glätten
        self.velocity_smoothing.append(self.velocity.z)
        if len(self.velocity_smoothing) > 5:
            self.velocity_smoothing.pop(0)
        smoothed_velocity = sum(self.velocity_smoothing) / len(self.velocity_smoothing)
        self.velocity.z = smoothed_velocity

        self.altitude_history.append(self.position.z)
        self.velocity_history.append(self.velocity.z)
        self.fuel_history.append(self.fuel)
        self.time_history.append(self.time_history[-1] + delta_time if self.time_history else delta_time)
        self.position_history.append((self.position.x, self.position.y, self.position.z))

        # Neue Höhenkontrolle
        if self.position.z > MAX_ALTITUDE or math.isnan(self.position.z):
            print(f"Warning: Altitude value unstable ({self.position.z}m), adjusting to {max(0, self.position.z - abs(self.velocity.z) * delta_time)}m")
            self.position.z = max(0, self.position.z - abs(self.velocity.z) * delta_time)
        if abs(self.velocity.z) > MAX_VELOCITY or math.isnan(self.velocity.z):
            print(f"Warning: Velocity value unstable ({self.velocity.z}m/s), capping to {MAX_VELOCITY * np.sign(self.velocity.z)}m/s")
            self.velocity.z = MAX_VELOCITY * (1 if self.velocity.z > 0 else -1)

    def calculate_drag(self):
        if self.position.z <= 0:
            return 0
        try:
            density = MARS_ATMOSPHERE_DENSITY_SEA_LEVEL * math.exp(-self.position.z / MARS_SCALE_HEIGHT)
            speed = self.velocity.magnitude()
            if speed > 0 and self.position.z < 5000:  # Stärkerer Drag in niedriger Höhe
                drag_force = 0.5 * density * speed ** 2 * DRAG_COEFFICIENT * CROSS_SECTIONAL_AREA * (5000 / (self.position.z + 1))
                drag_acc = -drag_force / STARSHIP_MASS * (self.velocity.z / speed)
            else:
                drag_force = 0.5 * density * speed ** 2 * DRAG_COEFFICIENT * CROSS_SECTIONAL_AREA
                drag_acc = -drag_force / STARSHIP_MASS * (self.velocity.z / speed)
            if math.isnan(drag_acc) or math.isinf(drag_acc):
                return 0
            return drag_acc
        except Exception as e:
            print(f"Error in drag calculation: {e}")
            return 0

    def calculate_centrifugal(self):
        try:
            radius = 3.3895e6  # Mars radius
            centrifugal_acc = (MARS_ROTATION_OMEGA ** 2) * radius * 1e-3
            if math.isnan(centrifugal_acc) or math.isinf(centrifugal_acc):
                return 0
            return centrifugal_acc
        except Exception as e:
            print(f"Error in centrifugal calculation: {e}")
            return 0

    def calculate_wind(self):
        try:
            current_time = self.time_history[-1] if self.time_history else 0
            wind_strength = WIND_MAX_ACCELERATION * (1 + min(self.position.z, 10000) / 10000)
            gust = math.sin(current_time * 0.5) * wind_strength * 0.5
            wind_x = random.uniform(-wind_strength, wind_strength) + gust
            wind_y = random.uniform(-wind_strength, wind_strength) + gust
            if math.isnan(wind_x) or math.isinf(wind_x):
                wind_x = 0
            if math.isnan(wind_y) or math.isinf(wind_y):
                wind_y = 0
            return Vector3D(wind_x, wind_y, 0)
        except Exception as e:
            print(f"Error in wind calculation: {e}")
            return Vector3D(0, 0, 0)

    def scan_and_navigate(self):
        try:
            scan_data = self.lidar.scan(self.position, self.velocity)
            backup_data = self.backup_lidar.scan(self.position, self.velocity)

            scan_data.distance = min(max(scan_data.distance, 0), MAX_ALTITUDE)
            backup_data.distance = min(max(backup_data.distance, 0), MAX_ALTITUDE)

            if len(self.altitude_history) >= 5:
                recent_altitudes = self.altitude_history[-5:]
                median_altitude = sorted(recent_altitudes)[2]  # Median statt Durchschnitt
                if abs(scan_data.distance - median_altitude) > 50:
                    scan_data = backup_data
                    print("Warning: Primary LIDAR outlier detected, using backup.")

            altitude = scan_data.distance
            velocity = scan_data.velocity

            if abs(altitude - self.position.z) > 100 or abs(velocity.z) > 500:
                velocity = self.backup_lidar.scan(self.position, self.velocity).velocity
                altitude = self.position.z
                print("Warning: Sensor fusion failed, using primary LIDAR and position.")

            self.velocity = velocity

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
                    self.thrust = min(STARSHIP_MASS * MARS_GRAVITY * 1.2, MAX_THRUST * (self.active_engines / 3))
            elif altitude < HOVER_ALTITUDE + 50:
                self.thrust = min(STARSHIP_MASS * MARS_GRAVITY * 1.0, MAX_THRUST * (self.active_engines / 3))
                print("Hovering at {:.1f}m to confirm landing zone...".format(altitude))
            else:
                self.thrust = min(STARSHIP_MASS * MARS_GRAVITY * 0.9, MAX_THRUST * (self.active_engines / 3))

            return altitude
        except Exception as e:
            print(f"Error in scan_and_navigate: {e}")
            self.thrust = STARSHIP_MASS * MARS_GRAVITY * 0.9
            return self.position.z

    def analyze_terrain(self, terrain_map):
        try:
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
        except Exception as e:
            print(f"Error in analyze_terrain: {e}")
            return None

    def adjust_trajectory(self, target_coords):
        try:
            delta = Vector3D(target_coords.x - self.position.x,
                             target_coords.y - self.position.y, 0)
            distance = delta.magnitude()

            target_velocity_z = -2 if self.position.z > 500 else -0.3
            thrust_adjustment = self.pid.compute(self.velocity.z, target_velocity_z)

            if math.isnan(thrust_adjustment) or math.isinf(thrust_adjustment) or abs(thrust_adjustment) > 10:
                thrust_adjustment = self.backup_pid.compute(self.velocity.z, target_velocity_z)
                print("Warning: Primary PID unstable, using backup PID.")
                if math.isnan(thrust_adjustment) or math.isinf(thrust_adjustment):
                    thrust_adjustment = 0
                    print("Warning: Both PIDs failed, using default thrust.")

            self.thrust = min(MAX_THRUST * (self.active_engines / 3),
                              STARSHIP_MASS * (MARS_GRAVITY + thrust_adjustment))

            if distance > 50:
                self.velocity.x = delta.x * 0.015
                self.velocity.y = delta.y * 0.015
            else:
                self.velocity.x *= 0.85
                self.velocity.y *= 0.85
        except Exception as e:
            print(f"Error in adjust_trajectory: {e}")
            self.thrust = STARSHIP_MASS * MARS_GRAVITY * 0.9

    def emergency_abort(self):
        print("Critical failure detected! Initiating emergency landing...")
        self.thrust = MAX_THRUST * (self.active_engines / 3)
        self.velocity.z = 10
        print("Ascending to safe altitude. Mission aborted.")
        return False

    def land(self):
        print("Initiating Starship Mars landing sequence (thrust-only)...")
        delta_time = 0.002  # Noch kleinere Schritte für Stabilität
        self.time_history.append(0)

        abort = False
        max_iterations = 500000  # Erhöht wegen kleinerem delta_time
        iterations = 0

        while self.position.z > 0 and not abort and iterations < max_iterations:
            try:
                iterations += 1
                altitude = self.scan_and_navigate()
                self.update_state(delta_time)

                if iterations % 500 == 0:  # Anpassung wegen kleinerem delta_time
                    print(
                        f"Altitude: {altitude:.1f}m, Velocity: {self.velocity.z:.1f}m/s, Fuel: {self.fuel:.1f}kg, Thrust: {self.thrust:.0f}N, Engines: {self.active_engines}")

                if self.fuel <= 0:
                    print("Out of fuel! Crash imminent.")
                    break
                if altitude < MIN_SAFE_ALTITUDE and not self.safe_zones:
                    abort = self.emergency_abort()
                    break
                if self.position.z <= 0:
                    if abs(self.velocity.z) <= SAFE_LANDING_SPEED:
                        print("Perfect landing achieved on Mars!")
                        self.plot_landing_data()
                    else:
                        print(f"Crash landing at {abs(self.velocity.z):.1f}m/s!")
                    break
            except Exception as e:
                print(f"Error in landing loop: {e}")
                abort = True

        if iterations >= max_iterations:
            print("Landing aborted: Maximum iterations reached")

        print(f"Final altitude: {self.position.z:.1f}m, Velocity: {self.velocity.z:.1f}m/s, Fuel: {self.fuel:.1f}kg")
        self.plot_landing_data()

    def plot_landing_data(self):
        try:
            stride = max(1, len(self.time_history) // 1000)
            time_data = self.time_history[::stride]
            altitude_data = self.altitude_history[::stride]
            velocity_data = self.velocity_history[::stride]
            fuel_data = self.fuel_history[::stride]
            position_data = self.position_history[::stride]

            fig = plt.figure(figsize=(12, 10))

            ax = fig.add_subplot(221, projection='3d')
            x, y, z = zip(*position_data)
            ax.plot(x, y, z, 'b-', label='Trajectory')
            ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', marker='o', label='Landing')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Starship Mars Landing Trajectory')
            ax.legend()

            ax2 = fig.add_subplot(222)
            ax2.plot(time_data, altitude_data, 'b-')
            ax2.set_ylabel('Altitude (m)')
            ax2.set_title('Altitude vs Time')
            ax2.grid(True)

            ax3 = fig.add_subplot(223)
            ax3.plot(time_data, velocity_data, 'r-')
            ax3.set_ylabel('Velocity (m/s)')
            ax3.set_xlabel('Time (s)')
            ax3.set_title('Velocity vs Time')
            ax3.grid(True)

            ax4 = fig.add_subplot(224)
            ax4.plot(time_data, fuel_data, 'g-')
            ax4.set_ylabel('Fuel (kg)')
            ax4.set_xlabel('Time (s)')
            ax4.set_title('Fuel vs Time')
            ax4.grid(True)

            plt.tight_layout()

            desktop_path = os.path.expanduser("~/Desktop")
            save_path = os.path.join(desktop_path, 'mars_landing_simulation_3d.png')

            plt.savefig(save_path)
            plt.close(fig)
            print(f"Landing data plot saved as '{save_path}'")
        except Exception as e:
            print(f"Error saving plot: {e}")


class NavigationDopplerLidar:
    def scan(self, position, current_velocity):
        try:
            distance = position.z + random.uniform(-0.1, 0.1)
            velocity_x = current_velocity.x + random.uniform(-0.005, 0.005)
            velocity_y = current_velocity.y + random.uniform(-0.005, 0.005)
            velocity_z = current_velocity.z + random.uniform(-0.005, 0.005)

            distance = max(0, min(distance, MAX_ALTITUDE))
            velocity_x = max(-MAX_VELOCITY, min(velocity_x, MAX_VELOCITY))
            velocity_y = max(-MAX_VELOCITY, min(velocity_y, MAX_VELOCITY))
            velocity_z = max(-MAX_VELOCITY, min(velocity_z, MAX_VELOCITY))

            data = LaserScanData(
                distance=distance,
                velocity=Vector3D(velocity_x, velocity_y, velocity_z)
            )
            return data
        except Exception as e:
            print(f"Error in LIDAR scan: {e}")
            return LaserScanData(
                distance=position.z,
                velocity=Vector3D(current_velocity.x, current_velocity.y, current_velocity.z)
            )


class HazardDetectionLidar:
    def scan(self, position, altitude):
        try:
            terrain_map = []
            scan_range = max(50, min(200, int(altitude / 1.5)))

            for x in range(-scan_range, scan_range + 1, 10):
                for y in range(-scan_range, scan_range + 1, 10):
                    rand = random.random()
                    if rand < 0.1:
                        roughness = random.uniform(0, 0.02)
                        obstacles = 0
                        dust_density = random.uniform(0, 0.1)
                    elif rand < 0.4:
                        roughness = random.uniform(0.1, 0.5)
                        obstacles = random.randint(1, 3)
                        dust_density = random.uniform(0.2, 0.5)
                    else:
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
        except Exception as e:
            print(f"Error in hazard detection: {e}")
            return [TerrainPatch(
                coordinates=Vector3D(position.x, position.y, 0),
                altitude=0,
                roughness=0.01,
                obstacles=0,
                size=60,
                dust_density=0.05
            )]


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.max_integral = 100

    def compute(self, current, target):
        try:
            error = target - current
            self.integral = max(-self.max_integral, min(self.integral + error, self.max_integral))
            derivative = error - self.previous_error
            self.previous_error = error
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            output = max(-10, min(output, 10))
            return output
        except Exception as e:
            print(f"Error in PID computation: {e}")
            return 0


if __name__ == "__main__":
    test_cases = [
        {"name": "Nominal", "fuel": 200000, "seed": 42, "noise_factor": 1},
        {"name": "Low Fuel", "fuel": 120000, "seed": 43, "noise_factor": 1},
        {"name": "Engine Failure", "fuel": 200000, "seed": 44, "noise_factor": 1},
        {"name": "High Wind", "fuel": 200000, "seed": 45, "noise_factor": 1},
        {"name": "High LIDAR Noise", "fuel": 200000, "seed": 46, "noise_factor": 2}
    ]

    for test in test_cases:
        print(f"\nRunning test case: {test['name']}")
        random.seed(test["seed"])

        original_scan = NavigationDopplerLidar.scan
        noise_factor = test["noise_factor"]

        def custom_scan(self, position, current_velocity):
            try:
                distance = position.z + random.uniform(-0.1 * noise_factor, 0.1 * noise_factor)
                velocity_x = current_velocity.x + random.uniform(-0.005 * noise_factor, 0.005 * noise_factor)
                velocity_y = current_velocity.y + random.uniform(-0.005 * noise_factor, 0.005 * noise_factor)
                velocity_z = current_velocity.z + random.uniform(-0.005 * noise_factor, 0.005 * noise_factor)

                distance = max(0, min(distance, MAX_ALTITUDE))
                velocity_x = max(-MAX_VELOCITY, min(velocity_x, MAX_VELOCITY))
                velocity_y = max(-MAX_VELOCITY, min(velocity_y, MAX_VELOCITY))
                velocity_z = max(-MAX_VELOCITY, min(velocity_z, MAX_VELOCITY))

                data = LaserScanData(
                    distance=distance,
                    velocity=Vector3D(velocity_x, velocity_y, velocity_z)
                )
                return data
            except Exception as e:
                print(f"Error in custom LIDAR scan: {e}")
                return LaserScanData(
                    distance=position.z,
                    velocity=Vector3D(current_velocity.x, current_velocity.y, current_velocity.z)
                )

        NavigationDopplerLidar.scan = custom_scan

        try:
            starship = Starship()
            starship.fuel = test["fuel"]
            starship.land()
        except Exception as e:
            print(f"Error in test case {test['name']}: {e}")
        finally:
            NavigationDopplerLidar.scan = original_scan
