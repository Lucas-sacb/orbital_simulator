"""
Example: Creating a simple Solar System (Sun, Earth, Mars)
You can add the following code inside the main() function in the orbital_simulator.py file to define these bodies:
"""
def main():
    # --- SCENARIO 3: Simple Solar System ---
    # Data is approximate. Units are kg, m, m/s.
    sun = Body(name="Sun", mass=1.989e30, position=[0, 0, 0], velocity=[0, 0, 0])
    earth = Body(name="Earth", mass=5.972e24, position=[1.496e11, 0, 0], velocity=[0, 29780, 0])
    mars = Body(name="Mars", mass=6.417e23, position=[2.279e11, 0, 0], velocity=[0, 24070, 0])

    # --- CHOOSE SCENARIO ---
    bodies_to_simulate = [sun, earth, mars]
    
    # For large-scale systems, you need a larger time step to see movement.
    # 86400 seconds = 1 day
    simulation_steps = 1000
    time_step = 86400 

    sim = Simulation(bodies_to_simulate)
    history = sim.run(steps=simulation_steps, dt=time_step)
    # ... rest of the code
