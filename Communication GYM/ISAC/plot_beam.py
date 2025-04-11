import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def beam_pattern(weights, d, angles):
    """
    Calculate and plot the beam pattern of an antenna array in polar coordinates.

    Parameters:
    weights : numpy.ndarray
        Beamforming weights (complex values).
    d : element distance / wavelength
    angles : numpy.ndarray
        Array of angles (in degrees) to compute the beam pattern.
    """
    # Convert angles from degrees to radians
    angles_rad = np.radians(angles)

    # Number of antenna elements
    N = len(weights)

    # Element index array
    n = np.arange(N)

    # Calculate the array factor for each angle
    array_factor = np.zeros_like(angles_rad, dtype=complex)
    for i, angle in enumerate(angles_rad):
        phase_shifts = -2j * np.pi * d * n * np.sin(angle)
        array_factor[i] = np.sum(weights.reshape(len(weights[0]),N) * np.exp(phase_shifts).reshape(1,N) )

    # Calculate the magnitude of the array factor (power pattern)
    power_pattern = np.abs(array_factor) ** 2 / np.max(np.abs(array_factor) ** 2)
    power_pattern_db = 10 * np.log10(power_pattern )  # Normalize and convert to dB

    return angles, power_pattern_db

def plot_beampattern(angles, power_pattern_db, polar):

    if polar:
        angles_rad = np.radians(angles)
        # Plot the beam pattern in polar coordinates
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(angles_rad, power_pattern_db)  # Use radians for polar plotting
        ax.set_theta_zero_location('N')  # Set the zero of the angle (theta=0) to the top of the plot
        ax.set_theta_direction(-1)  # Set the direction of increase of theta to counterclockwise
        ax.set_title('Beam Pattern in Polar Coordinates', va='bottom')  # Set the title and its position
        ax.grid(True)
        plt.show()
    else:
        plt.figure()
        plt.plot(angles, power_pattern_db)  # x-axis: angles, y-axis: dB
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Beam Pattern (dB)')
        plt.title('Beam Pattern vs. Angle')
        plt.grid(True)
        plt.show()

