# Code by Abasalt Bahrami - Feb 28, 2023


import datetime
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio import np, plt, sp
from diffractio import degrees, mm, nm, um
import cv2
import numpy as np
import matplotlib.pyplot as plt

# test for GitHub


# ------------ ------------ ------------ ------------ ------------ ------------
# GAUSSIAN BEAM + PHASE MASK + VORTEX BEAM
# Define a function to calculate the electric field of a Gaussian beam given its radial distance, propagation distance,
# beam waist size, wavenumber, and curvature radius

def gaussian_beam(r, z, BeamSize, k, R):
    return (BeamSize * np.exp(-r**2/BeamSize**2)) * np.exp(1j * (k * z + k * r**2/2 * R))


# Set the parameters for the Gaussian beam
BeamSize = 200e-6  # Beam waist size in meters
wavelegth = 433e-9  # Wavelength of the beam in meters
k = 2 * np.pi / (wavelegth)  # Wavenumber of the beam
gridsize = 500e-6  # Size of the grid in meters
R = np.inf  # Curvature radius of the beam (infinity for a plane wave)

# Plot the Gaussian beam intensity profile in polar coordinates
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
plt.subplot(1, 4, 3)
R, Z = np.meshgrid(np.linspace(-gridsize, gridsize, 100),
                   np.linspace(-gridsize, gridsize, 100))
intensity = np.abs(gaussian_beam(R, Z, BeamSize, k, R))**2
plt.imshow(intensity/np.max(intensity), cmap='gist_heat')
plt.xlabel('R (mm)')
plt.ylabel('Z (mm)')
plt.xticks([])
plt.yticks([])
plt.title("Gaussian Intensity")
plt.colorbar(shrink=0.15)

# Plot the Gaussian beam intensity profile in Cartesian coordinates
plt.subplot(1, 4, 2)
z = 0  # Position along the optical axis
R = np.inf  # Curvature radius of the beam (infinity for a plane wave)
X, Y = np.meshgrid(np.linspace(-gridsize, gridsize, 100),
                   np.linspace(-gridsize, gridsize, 100))
r = np.sqrt(X**2 + Y**2)
gaussian_beam = BeamSize * \
    np.exp(-r**2/BeamSize**2) * np.exp(1j * (k * z + k * r**2/2/R))
plt.imshow(np.abs(gaussian_beam**2/np.max(gaussian_beam**2)), cmap='gist_heat')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.xticks([])
plt.yticks([])
plt.title('Gaussain Intensity')
plt.colorbar(shrink=0.15)

# Vortex Intensity Profile -- Cartesian
plt.subplot(1, 4, 4)  # create a subplot for the vortex intensity profile
X, Y = np.meshgrid(np.linspace(-gridsize, gridsize, 100),
                   np.linspace(-gridsize, gridsize, 100))  # create a grid of x and y values
R = np.sqrt(X**2 + Y**2)  # calculate the radial distance from the center
phi = np.arctan2(Y, X)  # calculate the azimuthal angle
charge = 1  # set the charge of the vortex
# create a phase mask for the vortex
VortexMask = R**charge * np.exp(1j * charge * phi)
# apply the phase mask to the Gaussian beam
VortexBeam = VortexMask*gaussian_beam
# display the intensity profile of the vortex beam
plt.imshow(np.abs((VortexBeam/np.max(VortexBeam))**2), cmap='gist_heat')
plt.xlabel('X (mm)')  # set the x-axis label
plt.ylabel('Y (mm)')  # set the y-axis label
plt.xticks([])  # remove the x-axis ticks
plt.yticks([])  # remove the y-axis ticks
plt.title('Vortex Intensity')  # set the title of the plot
plt.text(5, 10, "charge = " + str(charge), color='white',
         fontsize=13)  # add text to the plot
plt.colorbar(shrink=0.15)  # add a colorbar to the plot


# Vortex Mask Phase
plt.subplot(1, 4, 1)
plt.imshow(np.angle(VortexMask)/np.pi, cmap='magma')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title("Phase mask")
plt.xticks([])
plt.yticks([])
cb = plt.colorbar(ticks=[-0.99, 0, 0.99], shrink=0.15)
cb.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
plt.show()


# Resize the mask to the desired size
resized_mask = cv2.resize(np.angle(VortexMask)/np.pi, (1920, 1152))
# Normalize the values to the range [0, 255] and convert to uint8 data type
normalized_mask = cv2.normalize(
    resized_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# Get the current date and time
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Set the filename with the timestamp
filename = f"/Users/abasaltbahrami/Downloads/vortex_{timestamp}.bmp"
# Save the mask as a grayscale bmp image with the timestamp in the filename
cv2.imwrite(filename, normalized_mask)


# ------------ ------------ ------------ ------------ ------------ ------------
# PHASE RETURDER
# Define a function to calculate the thickness of the phase returder

def calculate_thickness(phase_shift, wavelength, refractive_index):
    return (phase_shift * wavelength) / (2 * np.pi * refractive_index)


# Define the parameters for the phase returder
wavelength = 403e-9      # Wavelength of the wave in meters
refractive_index = 2.0   # Refractive index of the waveguide material

# Calculate the thicknesses for different phase shifts
phase_shifts = np.linspace(0, 10, 1000)
thicknesses = calculate_thickness(
    phase_shifts * np.pi, wavelength, refractive_index)

# Plot the thicknesses as a function of phase shift
plt.plot(phase_shifts, thicknesses * 1e6)  # Convert thicknesses to micrometers
plt.xlabel('Phase Shift (π)')
plt.ylabel('Thickness (μm)')
plt.title('Thickness vs. Phase Shift')

# Display the plot
plt.show()


# ------------ ------------ ------------ ------------ ------------ ------------
# Library "DIFFRACTIO"
# The website "https://diffractio.readthedocs.io/en/latest/source/tutorial/scalar_XY/sources_xy.html" provides a
# tutorial on how to generate scalar beam profiles using XY sources. To create a beam profile, you will
# need to define an XY source, specify the wavelength of the light, and choose a propagation distance.


# Define parameters for generating the beam profiles
M = 7
x0 = np.linspace(-1 * mm, 1 * mm, 512)
y0 = np.linspace(-1 * mm, 1 * mm, 512)
wavelength = 0.444 * um

# Create an XY source for the beam
u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)

# Create a figure to display the beam profiles
ID1 = plt.figure(figsize=(12, 6))
plt.suptitle("$Vortex beams$", fontsize=20)

# Generate M vortex beams and plot their intensities and phases
for m in range(M):
    # Create a vortex beam with the given parameters
    u.vortex_beam(A=1, r0=(0 * um, 0 * um), w0=400 * um, m=m)

    # Calculate the intensity and phase of the beam
    intensity = np.abs(u.u)**2
    phase = np.angle(u.u) / degrees
    phase[intensity < 0.005] = 0  # Set the phase to 0 for low-intensity pixels

    # Plot the intensity and phase of the beam
    title = "(%d)" % (m)
    plt.subplot(2, M, m + 1)
    plt.axis('off')
    plt.title(title)
    h1 = plt.imshow(intensity)
    h1.set_cmap("gist_heat")
    plt.subplot(2, M, m + M + 1)
    plt.axis('off')
    h2 = plt.imshow(phase)
    h2.set_cmap("twilight")

# Display the plot
plt.show()
