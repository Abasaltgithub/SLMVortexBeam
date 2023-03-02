# SLMVortexBeam
The code generates an intensity profile of a Gaussian beam, phase mask, and vortex beam in Cartesian and polar coordinates.
The diffractio library is used to calculate the electric field of the Gaussian beam. numpy, matplotlib, and cv2 libraries are used for the numerical calculations, visualization, and image processing.
The code defines a function gaussian_beam to calculate the electric field of a Gaussian beam given its radial distance, propagation distance, beam waist size, wavenumber, and curvature radius. Then it sets the parameters for the Gaussian beam, such as beam waist size, wavelength, wavenumber, and grid size.
The code then plots the Gaussian beam intensity profile in polar and Cartesian coordinates. Next, it calculates the radial distance and azimuthal angle and applies the phase mask for the vortex beam. Finally, it displays the intensity profile of the vortex beam with a color bar.
The code also resizes the mask to the desired size, normalizes the values to the range [0, 255], and converts them to uint8 data type.
The datetime library is imported but not used in the code. Following are few results:

![Unknown-6](https://user-images.githubusercontent.com/83898640/222328706-8cf5fdc2-dbe4-485f-87df-acb079232f14.png)
![Unknown-7](https://user-images.githubusercontent.com/83898640/222328726-c1e611cd-e2cc-4982-aff7-61e89b931be6.png)
![Unknown-8](https://user-images.githubusercontent.com/83898640/222328735-6645c5e8-9d31-455e-a32c-d64b8e3740d5.png)
