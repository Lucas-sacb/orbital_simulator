# solar_system_simulation.py AVG Execution time: 4 min
import numpy as np
import matplotlib.pyplot as plt

def setup_celestial_bodies():
    """
    Initializes the data for the Sun and planets.
    
    This function contains the initial conditions (mass, position, velocity)
    for each celestial body on the date 01/01/2019.
    Positions are in Astronomical Units (AU), velocities are in AU/day,
    and masses are in kilograms.

    Returns:
        list: A list of dictionaries, where each dictionary represents a celestial body.
    """
    # Data for celestial bodies as of 01/01/2019
    # Source: NASA's JPL Horizons System (approximated)
    celestial_bodies_data = [
        {'name': 'Sun',     'mass': 1.989e30, 'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]},
        {'name': 'Mercury', 'mass': 3.285e23, 'position': [-0.30035, -0.33763, -3.5458e-05], 'velocity': [0.015307, -0.017403, -0.0028264]},
        {'name': 'Venus',   'mass': 4.867e24, 'position': [-0.56277, 0.44506, 0.038583], 'velocity': [-0.012623, -0.015967, 0.00050938]},
        {'name': 'Earth',   'mass': 5.972e24, 'position': [-0.18795, 0.96517, -4.5855e-05], 'velocity': [-0.017353, -0.00338941, 4.8569e-07]},
        {'name': 'Mars',    'mass': 6.39e23,  'position': [1.0814, 0.97292, -0.0061494], 'velocity': [-0.0088277, 0.011597, 0.00045961]},
        {'name': 'Jupiter', 'mass': 1.898e27, 'position': [-2.1274, -4.9079, 0.06798], 'velocity': [0.006839, -0.0026486, -0.000142044]},
        {'name': 'Saturn',  'mass': 5.683e26, 'position': [1.9647, -9.866, 0.093293], 'velocity': [0.0051712, 0.0010708, -0.00022456]},
        {'name': 'Uranus',  'mass': 8.681e25, 'position': [17.014, 10.24, -0.18226], 'velocity': [-0.0020512, 0.0031833, 3.8336e-05]},
        {'name': 'Neptune', 'mass': 1.024e26, 'position': [28.983, -7.4855, -0.51387], 'velocity': [0.00077096, 0.0030558, -0.000080881]}
    ]
    return celestial_bodies_data

def normalize_data(bodies_data, mass_norm, time_norm):
    """
    Normalizes the physical units for the simulation.
    
    Positions are already in AU, so no normalization is needed (r_norm = 1).
    Masses are normalized to solar masses.
    Velocities are normalized based on the time unit (1 year).
    The gravitational constant G in these units becomes 4 * pi^2.

    Args:
        bodies_data (list): List of dictionaries for celestial bodies.
        mass_norm (float): The mass of the Sun in kg.
        time_norm (float): The number of days in a year.

    Returns:
        tuple: A tuple containing NumPy arrays for names, masses, positions, and velocities.
    """
    num_bodies = len(bodies_data)
    
    # Initialize NumPy arrays for efficiency
    names = np.array([body['name'] for body in bodies_data])
    masses = np.array([body['mass'] for body in bodies_data])
    positions = np.array([body['position'] for body in bodies_data])
    velocities = np.array([body['velocity'] for body in bodies_data])
    
    # Normalize units
    masses /= mass_norm
    velocities *= time_norm # r_norm is 1 AU, so it's omitted

    return names, masses, positions, velocities

def calculate_total_acceleration(body_index, current_positions, masses):
    """
    Calculates the total gravitational acceleration on a single body.

    This function computes the vector sum of the gravitational forces exerted by all
    other bodies, then divides by the body's mass (F=ma => a=F/m).
    The gravitational constant G is assumed to be 4*pi^2 in the chosen units.

    Args:
        body_index (int): The index of the body to calculate acceleration for.
        current_positions (np.ndarray): A (N, 3) array of positions for all N bodies.
        masses (np.ndarray): A (N,) array of masses for all N bodies.

    Returns:
        np.ndarray: A (3,) array representing the acceleration vector [ax, ay, az].
    """
    G_normalized = 4 * (np.pi**2)
    total_acceleration = np.zeros(3)
    
    for j in range(len(masses)):
        if j != body_index:
            # Vector pointing from body j to body i
            r_vector = current_positions[body_index] - current_positions[j]
            # Distance cubed
            distance_cubed = np.linalg.norm(r_vector)**3
            
            # Add acceleration component from body j
            total_acceleration += -G_normalized * masses[j] * r_vector / distance_cubed
            
    return total_acceleration

def run_simulation(masses, initial_positions, initial_velocities, time_step, iterations):
    """
    Runs the N-body simulation using the 4th-order Runge-Kutta (RK4) method.

    It iterates through time, updating the position and velocity of each planet
    based on the gravitational forces from all other bodies. The Sun is treated
    as a body and is also affected by the planets (though its movement is minimal).
    
    Note: The original code started the planet loop at index 1, effectively fixing
    the Sun. This version includes the Sun in the simulation for a more accurate,
    fully interactive N-body simulation, but its position remains at the barycenter
    if its initial velocity is zero.

    Args:
        masses (np.ndarray): Normalized masses.
        initial_positions (np.ndarray): Normalized initial positions.
        initial_velocities (np.ndarray): Normalized initial velocities.
        time_step (float): The time step for each iteration (in years).
        iterations (int): The total number of iterations to simulate.

    Returns:
        np.ndarray: A (num_bodies, iterations, 3) array storing the position history.
    """
    num_bodies = len(masses)
    
    # Make copies to avoid modifying the original initial conditions
    current_positions = np.copy(initial_positions)
    current_velocities = np.copy(initial_velocities)
    
    # Array to store the trajectory of each body
    position_history = np.zeros((num_bodies, iterations, 3))
    
    for k in range(iterations):
        # Store the current position at the beginning of the step
        position_history[:, k, :] = current_positions
        
        # We must store the state before updating any planet for this time step
        # This is because the acceleration on planet 'i' depends on the position of planet 'j'
        # at the start of the time step, not its intermediate RK4 position.
        positions_at_start_of_step = np.copy(current_positions)
        velocities_at_start_of_step = np.copy(current_velocities)
        
        # In this version, we calculate for all bodies, including the Sun.
        for i in range(num_bodies):
            # --- RK4 Integration Step ---
            # Get the state of the current body
            pos_i = positions_at_start_of_step[i]
            vel_i = velocities_at_start_of_step[i]

            # Calculate k1
            k1_v = calculate_total_acceleration(i, positions_at_start_of_step, masses)
            k1_r = vel_i

            # Calculate k2
            temp_pos_k2 = np.copy(positions_at_start_of_step)
            temp_pos_k2[i] = pos_i + 0.5 * time_step * k1_r
            k2_v = calculate_total_acceleration(i, temp_pos_k2, masses)
            k2_r = vel_i + 0.5 * time_step * k1_v

            # Calculate k3
            temp_pos_k3 = np.copy(positions_at_start_of_step)
            temp_pos_k3[i] = pos_i + 0.5 * time_step * k2_r
            k3_v = calculate_total_acceleration(i, temp_pos_k3, masses)
            k3_r = vel_i + 0.5 * time_step * k2_v

            # Calculate k4
            temp_pos_k4 = np.copy(positions_at_start_of_step)
            temp_pos_k4[i] = pos_i + time_step * k3_r
            k4_v = calculate_total_acceleration(i, temp_pos_k4, masses)
            k4_r = vel_i + time_step * k3_v

            # Update velocity and position for body i
            current_velocities[i] += (time_step / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            current_positions[i] += (time_step / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)

    return position_history

def plot_orbits(position_history, names):
    """
    Creates a 3D plot of the simulated orbits.

    Args:
        position_history (np.ndarray): The simulation results.
        names (np.ndarray): The names of the celestial bodies for the legend.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(len(names)):
        # Extract trajectory for each body
        x = position_history[i, :, 0]
        y = position_history[i, :, 1]
        z = position_history[i, :, 2]
        ax.plot(x, y, z, label=names[i])

    # Mark the starting point of Earth for reference
    ax.plot([position_history[3, 0, 0]], [position_history[3, 0, 1]], [position_history[3, 0, 2]], 'o', color='blue', label='Earth Start')
    
    # Set plot labels and title
    ax.set_xlabel('X-axis (AU)')
    ax.set_ylabel('Y-axis (AU)')
    ax.set_zlabel('Z-axis (AU)')
    ax.set_title('Solar System Orbits Simulation')
    
    # Setting z-limits can help visualize the ecliptic plane
    max_range = np.max(np.abs(position_history[1:, :, :])) # Max range of planets
    ax.set_zlim(-max_range*0.1, max_range*0.1) # Zoom in on the z-axis
    
    plt.legend()
    plt.show()

def main():
    """
    Main function to orchestrate the simulation.
    """
    # --- Simulation Parameters ---
    ITERATIONS = 60000  # Number of simulation steps
    TIME_STEP = 2e-3    # Time step in years. (h in original code)
    
    # --- Normalization Constants ---
    MASS_NORM = 1.989e30  # Mass of the Sun in kg
    TIME_NORM = 365.25    # Days in a year
    
    # 1. Load and structure initial data
    bodies_data = setup_celestial_bodies()
    
    # 2. Normalize data for simulation units
    names, masses, positions, velocities = normalize_data(bodies_data, MASS_NORM, TIME_NORM)
    
    # 3. Run the N-body simulation
    print("Starting simulation... (this may take a moment)")
    position_history = run_simulation(masses, positions, velocities, TIME_STEP, ITERATIONS)
    print("Simulation finished.")
    
    # 4. Plot the results
    print("Generating plot...")
    plot_orbits(position_history, names)

# --- Entry point of the script ---
if __name__ == "__main__":
    main()
