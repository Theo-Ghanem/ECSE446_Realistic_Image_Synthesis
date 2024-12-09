# Rendering Project

Welcome to my Rendering Project repository for ECSE 446: Realistic/Advanced Image Synthesis. This project contains four assignments (A1-A4) focusing on various aspects of realistic image rendering.

## Table of Contents
- [How to run](#how-to-run)
- [Assignment 1: Build Your Renderer](#assignment-1-build-your-renderer)
- [Assignment 2: Direct Illumination](#assignment-2-direct-illumination)
- [Assignment 3: Advanced Direct Illumination](#assignment-3-advanced-direct-illumination)
- [Assignment 4: Path Tracing](#assignment-4-path-tracing)



## How to run

1. **Development Setup**:
    - Use Conda to manage Python environments.
    - Create and activate a new Conda environment:
      ```bash
      $ conda create --name taichi_env python=3.10.14
      $ conda activate taichi_env
      $ pip install -e .
      ```
2. **Running Code**:
    - Run the main script:
      ```bash
      $ python ./interactive/A1.py
      ```

## Assignment 1: Build the Renderer

### Overview

In this assignment, I built a basic renderer using Python and the Taichi library. The main tasks included:

1. **Eye Ray Generation**: Implementing logic to generate rays from the camera.
2. **Ray-Triangle Intersection**: Implementing the Möller-Trumbore intersection algorithm.
3. **First Rendered Image**: Using the implemented functions to render my first image.


### [A1 Video Demonstration](https://youtube.com/shorts/O-RSmC8fWqw)


## Assignment 2: Direct Illumination


### Overview

This assignment extended the basic renderer to include direct illumination, pixel anti-aliasing, and progressive rendering.

1. **Pixel Anti-aliasing and Progressive Renderer**: Implemented jittered eye rays and a progressive renderer.
2. **General Direct Illumination**: Implemented environment light mapping and BRDFs for diffuse and Phong reflections.
3. **Importance Sampling**: Implemented uniform spherical importance sampling and BRDF importance sampling.

### [A2 Video Demonstration](https://youtube.com/shorts/DEtHLdLbfEM?feature=share)



## Assignment 3: Advanced Direct Illumination

### Overview

This assignment builds on top of Assignment 1 and includes more advanced direct illumination techniques, such as mesh lights, multiple importance sampling, and environment light importance sampling.

1. **Mesh Lights**: Implement mesh lights as physical objects in the scene with emissivity.
2. **BRDF Importance Sampling**: Modify the BRDF Importance Sampling direct illumination estimator to support mesh lights.
3. **Light Importance Sampling**: Implement mesh light importance sampling.
4. **Multiple Importance Sampling**: Combine BRDF and light sampling distributions using multiple importance sampling.
5. **Environment Light Importance Sampling**: (For ECSE 546 students) Implement environment light importance sampling and validate the sampling distributions.


### [A3 Video Demonstration](https://youtu.be/tyKrI06I0VY)



## Assignment 4: Path Tracing

### Overview

In this assignment, I will implement a path tracer, which is a Monte Carlo method for simulating global illumination. The path tracing algorithm computes realistic lighting by tracing the paths of rays through the scene, recursively bouncing between surfaces and light sources.

Key components of this assignment include:

1. **Implicit Path Tracing**: Implementing a 1-sample path Monte Carlo estimator for global illumination.
2. **Explicit Path Tracing**: Separating direct and indirect lighting contributions and calculating them using importance sampling.
3. **Russian Roulette**: Using probabilistic termination for recursive rays to reduce bias in the path tracing.
4. **Caustics**: Implementing refraction and Total Internal Reflection (TIR) to simulate caustics, focusing on the refraction of light at material interfaces.

### [A4 Video Demonstration](https://youtu.be/EpIBOo05Lo8)
