# Sublating Vortex Beams using Spatial Light Modulator (SLM)

This repository contains code for generating and visualizing intensity profiles of Gaussian beams, phase masks, and vortex beams in both Cartesian and polar coordinates. It utilizes the `diffractio` library for calculating the electric field of the Gaussian beam and relies on essential Python libraries like `numpy`, `matplotlib`, and `cv2` for numerical computations, visualization, and image processing. The primary objective of this code is to sublate vortex beams using a Spatial Light Modulator (SLM).

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)

## Introduction

Vortex beams, characterized by a spiral phase structure, are generated by imposing a phase mask on a Gaussian beam. This code performs the following tasks:

1. Calculates the electric field of a Gaussian beam given user-defined parameters.
2. Plots the Gaussian beam's intensity profile in both polar and Cartesian coordinates.
3. Applies a phase mask to create a vortex beam.
4. Visualizes the intensity profile of the vortex beam.

This README provides instructions on how to run the code and presents some example results.

## Dependencies

The code relies on the following Python libraries:

- `numpy`: For numerical calculations.
- `matplotlib`: For data visualization.
- `cv2` (OpenCV): For image processing.
- `diffractio`: For calculating the electric field of the Gaussian beam.

You can install these dependencies using `pip` or your preferred package manager:

```bash
pip install numpy matplotlib opencv-python diffractio
```

## Usage

To use the code, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Abasaltgithub/SLMVortexBeam.git
   ```

2. Navigate to the repository directory:

   ```bash
   cd Abasaltgithub/SLMVortexBeam
   ```

3. Run the Python script to generate and visualize vortex beams:

   ```bash
   python slm.py
   ```

4. Customize the parameters in the code as needed to experiment with different beam characteristics and phase masks.

## Results

The code generates intensity profiles of vortex beams and visualizes them. Some example results are provided in the repository for your reference.

Please feel free to explore the code and adapt it to your specific needs.

![Unknown-6](https://user-images.githubusercontent.com/83898640/222328706-8cf5fdc2-dbe4-485f-87df-acb079232f14.png)
![Unknown-7](https://user-images.githubusercontent.com/83898640/222328726-c1e611cd-e2cc-4982-aff7-61e89b931be6.png)
![Unknown-8](https://user-images.githubusercontent.com/83898640/222328735-6645c5e8-9d31-455e-a32c-d64b8e3740d5.png)
![Figure_1](https://user-images.githubusercontent.com/83898640/222332782-d5e056fc-453b-4575-8137-4d02511e8101.png)
