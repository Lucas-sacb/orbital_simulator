# N-Body Orbital Simulator

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A highly-extensible N-body gravitational simulator written in Python. It uses the Runge-Kutta 4th order (RK4) method to accurately calculate and visualize the trajectories of celestial bodies in 2D or 3D.

![Simulation Demo](https://i.imgur.com/8a5x3xG.png)
*(Example of the 3D Earth-Satellite system simulation)*

---

## Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Creating Your Own Scenarios](#creating-your-own-scenarios)
- [Code Structure](#code-structure)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About The Project

This project simulates the motion of multiple bodies under their mutual gravitational influence. It was refactored from an initial 2-body script into an object-oriented architecture, allowing it to simulate any number of bodies (N-body systems).

The physics is driven by **Newton's Law of Universal Gravitation**, and the trajectories are integrated over time using the **4th Order Runge-Kutta (RK4) numerical method**, which is known for its high accuracy in dynamical simulations.

Visualization is handled by `Matplotlib`, offering both 2D and 3D plots of the resulting orbits.

## Features

- **N-Body Simulation:** Easily simulate 2, 3, or more bodies.
- **Object-Oriented Design:** The code is modular and easy to extend, with `Body` and `Simulation` classes.
- **High-Precision Integrator:** Utilizes the Runge-Kutta 4 (RK4) method for stability and accuracy.
- **2D & 3D Visualization:** Choose how you want to view the calculated trajectories.
- **Configurable Scenarios:** Easily switch between different systems (e.g., Earth-Satellite, a 3-star system) directly in the code.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need Python 3 and `pip` installed on your system.
- **Python 3:** [python.org](https://www.python.org/downloads/)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    The project depends on `numpy` and `matplotlib`. Create a `requirements.txt` file with the following content:
    ```txt
    numpy
    matplotlib
    ```
    Then, install the dependencies using pip:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To start the simulation, run the main script from your terminal:

```sh
python orbital_simulator.py
