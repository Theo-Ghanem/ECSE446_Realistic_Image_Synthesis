from typing import List

import taichi as ti
import taichi.math as tm
import numpy as np

from .geometry import Geometry
from .materials import MaterialLibrary, Material

#TODO: Implement Uniform Sampling Methods
@ti.data_oriented
class UniformSampler:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction() -> tm.vec3:
        # Generate two random numbers
        xi1 = ti.random()  # Random value in [0, 1]
        xi2 = ti.random()  # Random value in [0, 1]

        # Compute the direction
        omega_z = 2 * xi1 - 1  # Random value in [-1, 1]
        r = tm.sqrt(1 - omega_z * omega_z)
        phi = 2 * tm.pi * xi2
        omega_x = r * tm.cos(phi)
        omega_y = r * tm.sin(phi)

        return tm.vec3(omega_x, omega_y, omega_z)


    @staticmethod
    @ti.func
    def evaluate_probability() -> float:
        return 1. / (4. * tm.pi) # 1 / (Area of the unit sphere: 4pi)

#TODO: Implement BRDF Sampling Methods
@ti.data_oriented
class BRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        alpha = material.Ns  # Specular coefficient
        xi1 = ti.random() # Random value in [0, 1]
        xi2 = ti.random() # Random value in [0, 1]

        omega_z = 0.0
        if alpha == 1.0:  # Diffuse surface
            omega_z = tm.sqrt(xi1)
        elif alpha > 1: # Glossy Phong surface
            omega_z = tm.pow(xi1, 1 /(alpha + 1))

        r = tm.sqrt(1 - omega_z * omega_z)
        phi = 2 * tm.pi * xi2
        omega_x = r * tm.cos(phi)
        omega_y = r * tm.sin(phi)

        omega = tm.vec3(omega_x, omega_y, omega_z)
        omega_r = reflect(w_o, normal)
        basis = tm.mat3(1.0)
        if alpha == 1.0:
            basis = ortho_frames(normal)
        elif alpha > 1:
            basis = ortho_frames(omega_r)

        return basis @ omega



    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> float:
        alpha = material.Ns
        pdf = 0.0
        if alpha == 1:
            # Diffuse surface
            pdf = (1 / tm.pi) * max(0.0, tm.dot(normal, w_i))
        elif alpha > 1:
            # Specular surface
            omega_r = reflect(w_o, normal)
            pdf = ((alpha + 1) / (2 * tm.pi)) * tm.pow(max(0.0, tm.dot(omega_r, w_i)), alpha)
        return pdf


    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        brdf = tm.vec3(0.0)
        # Diffuse component
        diffuse = material.Kd

        # Specular component (Phong)
        alpha = material.Ns
        omega_r = reflect(w_o, normal)
        specular = material.Kd * max(0.0, tm.dot(normal, w_i))

        if alpha == 1:
            brdf = diffuse
        elif alpha > 1:
            brdf = specular

        return brdf

    @staticmethod
    @ti.func
    def evaluate_brdf_factor(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3, pdf: float) -> tm.vec3:
        brdf_value = tm.vec3(0.0)
        # Diffuse component
        diffuse = material.Kd / tm.pi

        # Specular component (Phong)
        alpha = material.Ns
        omega_r = reflect(w_o, normal)
        specular = material.Ks * ((alpha + 1) / (2 * tm.pi)) * tm.pow(max(0.0, tm.dot(omega_r, w_i)), alpha)

        if alpha == 1:
            brdf_value = diffuse
        else:
            brdf_value = specular

        cos_theta_i = max(0.0, tm.dot(normal, w_i))

        brdf_factor = tm.vec3(0.0)
        if pdf > 0.0: # Ensure the pdf is not zero to avoid division by zero
            brdf_factor = brdf_value * cos_theta_i / pdf

        return brdf_factor

# Microfacet BRDF based on PBR 4th edition
# https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#
# XTODO: Implement Microfacet BRDF Methods
# 546 only deliverable
@ti.data_oriented
class MicrofacetBRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass


    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> float: 
        pass
        

    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass


'''
Ignore for now
'''
@ti.data_oriented
class MeshLightSampler:

    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        pass


    def get_emissive_triangle_indices(self) -> List[int]:
        pass


    @ti.kernel
    def compute_emissive_triangle_areas(self):
        pass

    @ti.func
    def compute_triangle_area_given_id(self, triangle_id: int) -> float:
        pass
        

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        pass


    @ti.kernel
    def compute_cdf(self):
        pass


    @ti.func
    def sample_emissive_triangle(self) -> int:
        pass

    @ti.func
    def evaluate_probability(self) -> float:
        pass


    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        pass



@ti.func
def ortho_frames(axis_of_alignment:tm.vec3) -> tm.mat3:

  random_vec = tm.normalize(tm.vec3([ti.random(), ti.random(), ti.random()]))

  x_axis = tm.cross(axis_of_alignment, random_vec)
  x_axis = tm.normalize(x_axis)

  y_axis = tm.cross(x_axis, axis_of_alignment)
  y_axis = tm.normalize(y_axis)


  result = tm.mat3([x_axis, y_axis, axis_of_alignment]).transpose()

  return result


@ti.func
def reflect(ray_direction:tm.vec3, normal: tm.vec3) -> tm.vec3:
    return  (2 * tm.dot(normal, ray_direction) * normal) - ray_direction