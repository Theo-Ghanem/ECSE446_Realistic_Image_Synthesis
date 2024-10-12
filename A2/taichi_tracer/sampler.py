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
        xi1 = ti.random()
        xi2 = ti.random()

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
        return BRDF.rotate_to_normal(omega, normal)


    @staticmethod
    @ti.func
    def rotate_to_normal(omega: tm.vec3, normal: tm.vec3) -> tm.vec3: #NOT SURE ABOUT THIS
        up = tm.vec3(0, 0, 1)
        if abs(normal.z) > 0.999:
            up = tm.vec3(0, 1, 0)
        tangent = tm.cross(up, normal).normalized()
        bitangent = tm.cross(normal, tangent)
        return omega.x * tangent + omega.y * bitangent + omega.z * normal


    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> float:
        alpha = material.Ns
        rho_d = material.Kd
        rho_s = material.Ks
        result = 0.0
        if alpha == 1:
            result = rho_d.norm() / tm.pi
        elif alpha > 1:
            omega_r = reflect(w_o, normal)
            result = ((rho_s.norm()*(alpha + 1)) / (2 * tm.pi)) * max(0.0, tm.pow(tm.dot(omega_r, w_i), alpha))
        return result


    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        result = tm.vec3(0.0)
        # Diffuse component
        diffuse = material.Kd / tm.pi

        # Specular component (Phong)
        alpha = material.Ns
        omega_r = reflect(w_o, normal)
        specular = material.Ks * ((alpha + 1) / (2 * tm.pi)) * tm.pow(max(0.0, tm.dot(omega_r, w_i)), alpha)

        if alpha == 1:
            result = diffuse
        else:
            result = specular

        return result

    @staticmethod
    @ti.func
    def evaluate_brdf_factor(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        # Diffuse component
        diffuse_brdf_factor = BRDF.evaluate_brdf(material, w_o, w_i, normal) * max(0.0, tm.dot(normal, w_i))

        # Specular component (Phong)
        alpha = material.Ns
        omega_r = reflect(w_o, normal)
        specular_brdf_factor = (material.Ks * ((alpha + 2) / (2 * tm.pi)) * tm.pow(max(0.0, tm.dot(omega_r, w_i)), alpha)
                                * max(0.0, tm.dot(normal, w_i)))

        return diffuse_brdf_factor + specular_brdf_factor

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
def ortho_frames(v_z: tm.vec3) -> tm.mat3:
    pass


@ti.func
def reflect(ray_direction:tm.vec3, normal: tm.vec3) -> tm.vec3:
    return  (2 * tm.dot(normal, ray_direction) * normal) - ray_direction