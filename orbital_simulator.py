# Orbital Simulator
#
# Author: E. Lucas Santos C. Babosa
# Updated and refactored by @lucas-sacb.
#
# A generic N-body orbital simulator using the Runge-Kutta 4th order method.
# This code is designed to be clear, efficient, and easily extensible.

import numpy as np
import matplotlib.pyplot as plt
from typing import List

# --- Physical Constants ---
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2

class Body:
    """
    Represents a celestial body with mass, position, and velocity.
    
    Attributes:
        name (str): The name of the body (e.g., 'Earth').
        mass (float): The mass of the body in kilograms (kg).
        position (np.ndarray): A 3D vector for the body's position in meters (m).
        velocity (np.ndarray): A 3D vector for the body's velocity in meters/second (m/s).
    """
    def __init__(self, name: str, mass: float, position: List[float], velocity: List[float]):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

class Simulation:
    """
    Manages the N-body simulation, including the physics calculations and time integration.
    
    Attributes:
        bodies (List[Body]): A list of all Body objects in the simulation.
    """
    def __init__(self, bodies: List[Body]):
        self.bodies = bodies

    def _calculate_acceleration(self, body_positions: np.ndarray) -> np.ndarray:
        """
        Calculates the net acceleration for each body due to gravitational forces from all other bodies.
        
        Args:
            body_positions (np.ndarray): An (N, 3) array of current positions for all N bodies.
        
        Returns:
            np.ndarray: An (N, 3) array of accelerations for each body.
        """
        num_bodies = len(self.bodies)
        accelerations = np.zeros((num_bodies, 3))
        
        for i in range(num_bodies):
            for j in range(num_bodies):
                if i == j:
                    continue  # A body does not exert a force on itself
                
                # Vector pointing from body i to body j
                r_vector = body_positions[j] - body_positions[i]
                
                # Distance between the bodies
                distance = np.linalg.norm(r_vector)
                
                if distance == 0:
                    continue # Avoid division by zero if bodies are at the same position
                
                # Gravitational force magnitude: F = G * (m1*m2) / r^2
                # Acceleration of body i: a = F / m_i = G * m_j / r^2
                # Acceleration vector: a * (r_vector / distance)
                acceleration_magnitude = GRAVITATIONAL_CONSTANT * self.bodies[j].mass / distance**3
                accelerations[i] += acceleration_magnitude * r_vector
                
        return accelerations

    def _run_rk4_step(self, dt: float):
        """
        Performs a single time step update using the 4th Order Runge-Kutta method.
        """
        num_bodies = len(self.bodies)
        
        # Get initial positions and velocities as NumPy arrays for vectorized operations
        initial_positions = np.array([body.position for body in self.bodies])
        initial_velocities = np.array([body.velocity for body in self.bodies])

        # k1: evaluation at the start of the interval
        k1_v = self._calculate_acceleration(initial_positions)
        k1_r = initial_velocities

        # k2: evaluation at the midpoint
        k2_v = self._calculate_acceleration(initial_positions + k1_r * dt / 2)
        k2_r = initial_velocities + k1_v * dt / 2
        
        # k3: another evaluation at the midpoint
        k3_v = self._calculate_acceleration(initial_positions + k2_r * dt / 2)
        k3_r = initial_velocities + k2_v * dt / 2
        
        # k4: evaluation at the end of the interval
        k4_v = self._calculate_acceleration(initial_positions + k3_r * dt)
        k4_r = initial_velocities + k3_v * dt

        # Update position and velocity for each body
        for i, body in enumerate(self.bodies):
            body.velocity += (dt / 6) * (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i])
            body.position += (dt / 6) * (k1_r[i] + 2*k2_r[i] + 2*k3_r[i] + k4_r[i])

    def run(self, steps: int, dt: float) -> dict:
        """
        Executes the full simulation.
        
        Args:
            steps (int): The total number of simulation steps to run.
            dt (float): The time delta for each step in seconds.
            
        Returns:
            dict: A dictionary containing the position history for each body.
        """
        history = {body.name: np.zeros((steps, 3)) for body in self.bodies}
        
        print(f"Running simulation for {steps} steps with dt = {dt}s...")
        for step in range(steps):
            for i, body in enumerate(self.bodies):
                history[body.name][step] = body.position
            self._run_rk4_step(dt)
        print("Simulation complete.")
        return history

def plot_trajectories(history: dict, plot_3d: bool = True):
    """Plots the trajectories from the simulation history."""
    num_bodies = len(history)
    fig = plt.figure(figsize=(10, 8))
    
    if plot_3d:
        ax = fig.add_subplot(111, projection='3d')
        for name, positions in history.items():
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=name)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"3D Orbital Simulation ({num_bodies} Bodies)")
    else:
        ax = fig.add_subplot(111)
        for name, positions in history.items():
            ax.plot(positions[:, 0], positions[:, 1], label=name)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"2D Orbital Simulation ({num_bodies} Bodies)")
    
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def main():
    """Main function to set up and run a simulation scenario."""
    
    # --- SCENARIO 1: Earth and a Satellite (similar to original code) ---
    # Realistic distances and speeds for Low Earth Orbit (LEO)
    earth = Body(name="Earth", mass=5.972e24, position=[0, 0, 0], velocity=[0, 0, 0])
    # Satellite in LEO, approx 400km altitude, speed approx 7.7 km/s
    satellite = Body(name="Satellite", mass=1000, position=[6.771e6, 0, 0], velocity=[0, 7700, 1500])
    
    # --- SCENARIO 2: A simple 3-Body System (for demonstration) ---
    star1 = Body(name="Star A", mass=1.989e30, position=[0, 0, 0], velocity=[0, 0, 0])
    star2 = Body(name="Star B", mass=1.989e30, position=[1.5e11, 0, 0], velocity=[0, 21000, 0])
    planet = Body(name="Planet", mass=5.972e24, position=[0.75e11, 0.5e11, 0], velocity=[-10000, 15000, 0])

    # --- CHOOSE SCENARIO ---
    # To run the Earth-Satellite system, use this line:
    bodies_to_simulate = [earth, satellite]
    simulation_steps = 10000
    time_step = 60 # seconds
    
    # To run the 3-Body system, uncomment these lines:
    # bodies_to_simulate = [star1, star2, planet]
    # simulation_steps = 50000
    # time_step = 3600 # 1 hour

    sim = Simulation(bodies_to_simulate)
    history = sim.run(steps=simulation_steps, dt=time_step)
    
    # --- Ask user for plot type ---
    while True:
        choice = input("Choose plot type (2D or 3D): ").strip().upper()
        if choice == "2D":
            plot_trajectories(history, plot_3d=False)
            break
        elif choice == "3D":
            plot_trajectories(history, plot_3d=True)
            break
        else:
            print("Invalid choice. Please enter '2D' or '3D'.")

if __name__ == "__main__":
    main()
