import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
import logging
import time

# Robuste Pfadverwaltung für Logging
log_dir = "./logs"
log_file = os.path.join(log_dir, "mars_mission_log.txt")
try:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
except (FileNotFoundError, PermissionError) as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.warning(f"Could not create log file {log_file}. Logging to stdout. Error: {e}")

# Fix für MacOS plot saving
plt.switch_backend('Agg')

# Constants
MARS_GRAVITY = 3.71
MARS_ROTATION_OMEGA = 7.083e-5
MARS_ATMOSPHERE_DENSITY_SEA_LEVEL = 0.02
MARS_SCALE_HEIGHT = 11000
DRAG_COEFFICIENT = 0.75
CROSS_SECTIONAL_AREA = 100
SPECIFIC_IMPULSE = 350
G0 = 9.81
STARSHIP_MASS = 238000  # Increased dry mass with payload
MAX_THRUST = 7500000
MIN_THRUST = 2250000
ENGINE_STARTUP_DELAY = 0.1
MIN_FUEL_RESERVE = 100000  # Increased for emergency maneuvers
SAFE_LANDING_SPEED = 0.5
MIN_SAFE_ALTITUDE = 50
WIND_MAX_ACCELERATION = 0.05
HOVER_ALTITUDE = 100
MAX_ALTITUDE = 500000  # 500 km orbit (not used)
ORBIT_VELOCITY = 3319  # Approx. orbital velocity at 500 km (not used)
MAX_VELOCITY = 4000
MAX_FUEL = 1200000  # For Mars-to-Earth return

class RefuelingStation:
    def __init__(self, solar_power=100000, reactor_power=0):
        self.solar_power = solar_power  # kW
        self.reactor_power = reactor_power  # kW
        self.total_power = solar_power + reactor_power
        # Production rate: assuming 75 kWh per kg of fuel (Sabatier + Electrolysis)
        self.production_rate = self.total_power / 75  # kg/day
        self.production_efficiency = 0.9  # 90% efficiency to account for losses
        logging.info(f"Initialized RefuelingStation with {self.total_power:.1f} kW power, "
                     f"production rate: {self.production_rate:.2f} kg/day")

    def produce_fuel(self, fuel_needed):
        """Calculate days required to produce the needed fuel, accounting for efficiency."""
        days_required = fuel_needed / (self.production_rate * self.production_efficiency)
        fuel_produced = fuel_needed
        return fuel_produced, days_required  # kg, days

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

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

class Starship:
    def __init__(self, mode='land', fuel=200000):
        self.mode = mode
        if mode == 'land':
            self.position = Vector3D(0, 0, 10000)
            self.velocity = Vector3D(0, 0, -50)
            self.fuel = fuel
        else:  # launch
            self.position = Vector3D(0, 0, 0)
            self.velocity = Vector3D(0, 0, 0)
            self.fuel = fuel
        self.thrust = 0
        self.active_engines = 3
        self.engine_delay_timer = 0
        self.pid = PIDController(kp=100.0, ki=0.1, kd=20.0)
        self.backup_pid = PIDController(kp=50.0, ki=0.05, kd=10.0)
        self.altitude_history = [self.position.z]
        self.velocity_history = [self.velocity.z]
        self.fuel_history = [self.fuel]
        self.time_history = [0]
        self.position_history = [(self.position.x, self.position.y, self.position.z)]
        self.velocity_smoothing = [self.velocity.z]
        self.warning_counter = 0
        self.pitch_history = []
        self.refuel_history = []  # Track refueling progress
        logging.info(f"Initialized Starship in {mode} mode with {fuel:.1f} kg fuel")

    def refuel_on_mars(self, refueling_station):
        """Simulate refueling on Mars to full capacity for return flight."""
        fuel_needed = MAX_FUEL - self.fuel
        if fuel_needed <= 0:
            logging.info("Starship already fully fueled.")
            print("Starship already fully fueled.")
            self.refuel_history.append((self.time_history[-1], self.fuel))
            return self.fuel
        fuel_produced, days_required = refueling_station.produce_fuel(fuel_needed)
        self.fuel += fuel_produced
        logging.info(f"Refueled {fuel_produced:.1f} kg on Mars in {days_required:.1f} days. "
                     f"Starship ready for return flight with {self.fuel:.1f} kg fuel.")
        print(f"Refueled {fuel_produced:.1f} kg on Mars in {days_required:.1f} days. "
              f"Starship ready for return flight with {self.fuel:.1f} kg fuel.")
        self.refuel_history.append((self.time_history[-1], self.fuel))
        return self.fuel

    def update_state(self, delta_time):
        total_mass = STARSHIP_MASS + self.fuel
        if self.fuel <= MIN_FUEL_RESERVE:
            self.fuel = MIN_FUEL_RESERVE
            self.thrust = 0
            logging.warning(f"Fuel at reserve level: {self.fuel:.1f} kg. Thrust disabled.")

        current_max_thrust = MAX_THRUST * (self.active_engines / 3)
        current_min_thrust = MIN_THRUST * (self.active_engines / 3)

        if self.thrust > 0 and self.engine_delay_timer > 0:
            self.engine_delay_timer -= delta_time
            thrust_acc = Vector3D(0, 0, 0)
        else:
            if self.mode == 'land':
                if self.position.z > 1000:
                    target_velocity_z = -20  # More aggressive descent
                elif self.position.z > 50:
                    target_velocity_z = -5
                elif self.position.z > 1:
                    target_velocity_z = -0.5
                else:
                    target_velocity_z = 0

                thrust_adjustment = self.pid.compute(self.velocity.z, target_velocity_z)
                if math.isnan(thrust_adjustment) or math.isinf(thrust_adjustment) or abs(thrust_adjustment) > 50:
                    thrust_adjustment = self.backup_pid.compute(self.velocity.z, target_velocity_z)
                    if self.warning_counter % 100 == 0:
                        logging.warning("Warning: Primary PID unstable, using backup PID.")
                    self.warning_counter += 1
                    if math.isnan(thrust_adjustment) or math.isinf(thrust_adjustment):
                        thrust_adjustment = 0

                self.thrust = max(0, min(total_mass * (MARS_GRAVITY + thrust_adjustment), current_max_thrust))

                thrust_acc = Vector3D(0, 0, self.thrust / total_mass)
                self.pitch_history.append(90)
            else:
                # Not used in this simulation
                self.thrust = current_max_thrust
                if self.position.z < 500:
                    pitch = 90
                else:
                    pitch = max(0, 90 - ((self.position.z - 500) / (200000 - 500)) * 90)
                pitch_rad = math.radians(pitch)
                thrust_z = self.thrust * math.sin(pitch_rad)
                thrust_x = self.thrust * math.cos(pitch_rad)
                thrust_acc = Vector3D(thrust_x / total_mass, 0, thrust_z / total_mass)
                self.pitch_history.append(pitch)

        gravity_acc = Vector3D(0, 0, -MARS_GRAVITY)
        drag_acc = self.calculate_drag()
        centrifugal_acc = Vector3D(0, 0, self.calculate_centrifugal())
        wind_acc = self.calculate_wind()

        acceleration = thrust_acc + gravity_acc + drag_acc + centrifugal_acc + wind_acc
        self.velocity = self.velocity + acceleration * delta_time
        self.position = self.position + self.velocity * delta_time

        if self.thrust > 0 and self.engine_delay_timer <= 0:
            mass_flow_rate = self.thrust / (SPECIFIC_IMPULSE * G0)
            fuel_used = mass_flow_rate * delta_time
            fuel_used = min(fuel_used, self.fuel - MIN_FUEL_RESERVE)
            self.fuel -= fuel_used
            if iterations % 100 == 0:  # More frequent logging
                logging.debug(f"Fuel used this step: {fuel_used:.1f} kg, Mass flow rate: {mass_flow_rate:.1f} kg/s, Remaining fuel: {self.fuel:.1f} kg")

        self.velocity_smoothing.append(self.velocity.z)
        if len(self.velocity_smoothing) > 3:
            self.velocity_smoothing.pop(0)
        smoothed_velocity_z = sum(self.velocity_smoothing) / len(self.velocity_smoothing)
        self.velocity.z = smoothed_velocity_z

        if self.mode == 'land':
            self.position.z = max(0, self.position.z)
        elif self.mode == 'launch' and self.position.z < 0:
            logging.warning(f"Negative altitude detected: {self.position.z:.1f}m. Resetting to 0.")
            self.position.z = 0
            self.velocity.z = max(0, self.velocity.z)

        self.altitude_history.append(self.position.z)
        self.velocity_history.append(self.velocity.z)
        self.fuel_history.append(self.fuel)
        self.time_history.append(self.time_history[-1] + delta_time)
        self.position_history.append((self.position.x, self.position.y, self.position.z))

        if self.velocity.magnitude() > MAX_VELOCITY or math.isnan(self.velocity.z):
            if self.warning_counter % 100 == 0:
                logging.warning(f"Warning: Velocity unstable ({self.velocity.magnitude()}m/s), capping.")
            self.warning_counter += 1
            scale = MAX_VELOCITY / self.velocity.magnitude()
            self.velocity = self.velocity * scale

    def calculate_drag(self):
        if self.mode == 'launch':
            return Vector3D(0, 0, 0)
        try:
            density = MARS_ATMOSPHERE_DENSITY_SEA_LEVEL * math.exp(-self.position.z / MARS_SCALE_HEIGHT)
            speed = self.velocity.magnitude()
            if speed == 0:
                return Vector3D(0, 0, 0)
            drag_force = 0.5 * density * speed ** 2 * DRAG_COEFFICIENT * CROSS_SECTIONAL_AREA
            drag_acc_magnitude = -drag_force / (STARSHIP_MASS + self.fuel)
            drag_acc = self.velocity * (drag_acc_magnitude / speed)
            if math.isnan(drag_acc.x) or math.isinf(drag_acc.x):
                return Vector3D(0, 0, 0)
            return drag_acc
        except Exception as e:
            logging.error(f"Error in drag calculation: {e}")
            return Vector3D(0, 0, 0)

    def calculate_centrifugal(self):
        try:
            radius = 3.3895e6
            centrifugal_acc = (MARS_ROTATION_OMEGA ** 2) * radius * 1e-3
            if math.isnan(centrifugal_acc) or math.isinf(centrifugal_acc):
                return 0
            return centrifugal_acc
        except Exception as e:
            logging.error(f"Error in centrifugal calculation: {e}")
            return 0

    def calculate_wind(self):
        if self.mode == 'land':
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
                logging.error(f"Error in wind calculation: {e}")
                return Vector3D(0, 0, 0)
        else:
            return Vector3D(0, 0, 0)

    def emergency_abort(self):
        logging.info("Critical failure detected! Initiating emergency procedure...")
        print("Critical failure detected! Initiating emergency procedure...")
        sys.stdout.flush()
        if self.mode == 'land':
            self.thrust = MAX_THRUST * (self.active_engines / 3)
            self.velocity.z = 10
            logging.info("Ascending to safe altitude. Landing aborted.")
            print("Ascending to safe altitude. Landing aborted.")
        else:
            self.thrust = 0
            logging.info("Launch aborted due to critical failure.")
            print("Launch aborted due to critical failure.")
        sys.stdout.flush()
        return True

    def simulate(self):
        global iterations
        logging.info(f"Initiating Starship Mars {self.mode} sequence...")
        print(f"Initiating Starship Mars {self.mode} sequence...")
        sys.stdout.flush()
        delta_time = 0.01
        self.time_history = [0]
        abort = False
        max_iterations = 100000
        iterations = 0

        # Initialize refueling station for Mars
        refueling_station = RefuelingStation(solar_power=100000, reactor_power=0)

        while iterations < max_iterations and not abort:
            start_time = time.time()
            try:
                iterations += 1
                self.update_state(delta_time)

                if iterations % 500 == 0:
                    horizontal_velocity = (self.velocity.x ** 2 + self.velocity.y ** 2) ** 0.5
                    pitch = self.pitch_history[-1] if self.pitch_history else 90
                    logging.info(
                        f"Altitude: {self.position.z:.1f}m, Vertical Velocity: {self.velocity.z:.1f}m/s, "
                        f"Horizontal Velocity: {horizontal_velocity:.1f}m/s, Fuel: {self.fuel:.1f}kg, "
                        f"Thrust: {self.thrust:.0f}N, Engines: {self.active_engines}, Pitch: {pitch:.1f}°")
                    print(
                        f"Altitude: {self.position.z:.1f}m, Vertical Velocity: {self.velocity.z:.1f}m/s, "
                        f"Horizontal Velocity: {horizontal_velocity:.1f}m/s, Fuel: {self.fuel:.1f}kg, "
                        f"Thrust: {self.thrust:.0f}N, Engines: {self.active_engines}, Pitch: {pitch:.1f}°")
                    sys.stdout.flush()

                if self.fuel <= MIN_FUEL_RESERVE:
                    logging.info("Fuel at minimum reserve! Mission failed.")
                    print("Fuel at minimum reserve! Mission failed.")
                    sys.stdout.flush()
                    break

                if self.mode == 'land':
                    if self.position.z < 1:
                        if abs(self.velocity.z) <= SAFE_LANDING_SPEED:
                            logging.info("Perfect landing achieved on Mars!")
                            print("Perfect landing achieved on Mars!")
                            sys.stdout.flush()
                            # Simulate refueling after landing
                            self.refuel_on_mars(refueling_station)
                            self.plot_mission_data()
                            break
                        else:
                            logging.info(f"Crash landing at {abs(self.velocity.z):.1f}m/s!")
                            print(f"Crash landing at {abs(self.velocity.z):.1f}m/s!")
                            sys.stdout.flush()
                            break

            except Exception as e:
                logging.error(f"Error in simulation loop: {e}")
                print(f"Error in simulation loop: {e}")
                sys.stdout.flush()
                abort = self.emergency_abort()
            finally:
                logging.info(f"Iteration {iterations} took {time.time() - start_time:.4f} seconds")

        if iterations >= max_iterations:
            logging.info("Mission aborted: Maximum iterations reached")
            print("Mission aborted: Maximum iterations reached")
            sys.stdout.flush()

        horizontal_velocity = (self.velocity.x ** 2 + self.velocity.y ** 2) ** 0.5
        logging.info(f"Final altitude: {self.position.z:.1f}m, Vertical Velocity: {self.velocity.z:.1f}m/s, "
                     f"Horizontal Velocity: {horizontal_velocity:.1f}m/s, Fuel: {self.fuel:.1f}kg")
        print(f"Final altitude: {self.position.z:.1f}m, Vertical Velocity: {self.velocity.z:.1f}m/s, "
              f"Horizontal Velocity: {horizontal_velocity:.1f}m/s, Fuel: {self.fuel:.1f}kg")
        sys.stdout.flush()
        if self.mode == 'land' and self.position.z < 1 and abs(self.velocity.z) <= SAFE_LANDING_SPEED:
            logging.info("Simulation complete: Starship landed and refueled on Mars.")
            print("Simulation complete: Starship landed and refueled on Mars.")
        else:
            self.plot_mission_data()

    def plot_mission_data(self):
        try:
            min_length = min(len(self.time_history), len(self.altitude_history), len(self.fuel_history))
            stride = max(1, min_length // 1000)
            time_data = self.time_history[:min_length:stride]
            altitude_data = self.altitude_history[:min_length:stride]
            velocity_data = self.velocity_history[:min_length:stride]
            fuel_data = self.fuel_history[:min_length:stride]
            position_data = self.position_history[:min_length:stride]

            # Prepare refueling data
            refuel_times = [t for t, _ in self.refuel_history]
            refuel_amounts = [f for _, f in self.refuel_history]
            if not refuel_times:
                refuel_times = [0]
                refuel_amounts = [0]

            fig = plt.figure(figsize=(15, 12))

            ax = fig.add_subplot(221, projection='3d')
            x, y, z = zip(*position_data)
            ax.plot(x, y, z, 'b-', label='Trajectory')
            ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', marker='o', label='Final Position')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Starship Mars {self.mode.capitalize()} Trajectory')
            ax.legend()

            ax2 = fig.add_subplot(222)
            ax2.plot(time_data, altitude_data, 'b-')
            ax2.set_ylabel('Altitude (m)')
            ax2.set_xlabel('Time (s)')
            ax2.set_title('Altitude vs Time')
            ax2.grid(True)

            ax3 = fig.add_subplot(223)
            ax3.plot(time_data, velocity_data, 'r-')
            ax3.set_ylabel('Vertical Velocity (m/s)')
            ax3.set_xlabel('Time (s)')
            ax3.set_title('Vertical Velocity vs Time')
            ax3.grid(True)

            ax4 = fig.add_subplot(224)
            ax4.plot(time_data, fuel_data, 'g-', label='Fuel Level')
            ax4.plot(refuel_times, refuel_amounts, 'm*', label='Refueling Events')
            ax4.set_ylabel('Fuel (kg)')
            ax4.set_xlabel('Time (s)')
            ax4.set_title('Fuel and Refueling vs Time')
            ax4.grid(True)
            ax4.legend()

            plt.tight_layout()

            desktop_path = os.path.expanduser("~/Desktop")
            save_path = os.path.join(desktop_path, f'mars_{self.mode}_simulation_3d.png')
            try:
                if not os.path.exists(desktop_path):
                    os.makedirs(desktop_path)
                plt.savefig(save_path)
            except (FileNotFoundError, PermissionError) as e:
                logging.warning(f"Could not save plot to {save_path}. Saving to current directory instead. Error: {e}")
                plt.savefig(f'mars_{self.mode}_simulation_3d.png')
            plt.close(fig)
            logging.info(f"Mission data plot saved as '{save_path}' or 'mars_{self.mode}_simulation_3d.png'")
            print(f"Mission data plot saved as '{save_path}' or 'mars_{self.mode}_simulation_3d.png'")
            sys.stdout.flush()
        except Exception as e:
            logging.error(f"Error saving plot: {e}")
            print(f"Error saving plot: {e}")
            sys.stdout.flush()

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
            if abs(error) < 0.1:
                self.integral = 0
            else:
                self.integral = max(-self.max_integral, min(self.integral + error, self.max_integral))
            derivative = error - self.previous_error
            self.previous_error = error
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            output = max(-50, min(output, 50))
            return output
        except Exception as e:
            logging.error(f"Error in PID computation: {e}")
            return 0

if __name__ == "__main__":
    logging.info("Starting Mars mission simulation with updated code (2025-05-06)")
    test_cases = [
        {"name": "Landing Nominal", "mode": "land", "fuel": 600000, "seed": 46}
    ]

    for test in test_cases:
        print(f"\nRunning test case: {test['name']}")
        logging.info(f"\nRunning test case: {test['name']}")
        sys.stdout.flush()
        random.seed(test["seed"])

        try:
            starship = Starship(mode=test["mode"], fuel=test["fuel"])
            starship.simulate()
        except Exception as e:
            logging.error(f"Error in test case {test['name']}: {e}")
            print(f"Error in test case {test['name']}: {e}")
            sys.stdout.flush()

    sys.stdout.flush()
