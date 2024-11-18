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
        xi1 = ti.random()  # Random value in [0, 1]
        xi2 = ti.random()  # Random value in [0, 1]

        omega_z = 0.0
        if alpha == 1.0:  # Diffuse surface
            omega_z = tm.sqrt(xi1)
        elif alpha > 1:  # Glossy Phong surface
            omega_z = tm.pow(xi1, 1 / (alpha + 1))

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
        specular = material.Kd * max(0.0, tm.dot(normal, w_i))

        if alpha == 1:
            brdf = diffuse
        elif alpha > 1:
            brdf = specular

        return brdf

    @staticmethod
    @ti.func
    def evaluate_brdf_factor(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3, pdf: float) -> tm.vec3:  # Used for Uniform

        # Diffuse component
        diffuse = material.Kd / tm.pi

        # Specular component (Phong)
        alpha = material.Ns
        omega_r = reflect(w_o, normal)
        specular = material.Ks * ((alpha + 1) / (2 * tm.pi)) * tm.pow(max(0.0, tm.dot(omega_r, w_i)), alpha)

        brdf_value = tm.vec3(0.0)
        if alpha == 1:
            brdf_value = diffuse
        else:
            brdf_value = specular

        brdf_factor = tm.vec3(0.0)
        if pdf > 0.0:  # Ensure the pdf is not zero to avoid division by zero
            brdf_factor = brdf_value * max(0.0, tm.dot(normal, w_i)) / pdf

        return brdf_factor

# Microfacet BRDF based on PBR 4th edition
# https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#
# xTODO: Implement Microfacet BRDF Methods
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


@ti.data_oriented
class MeshLightSampler:

    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        # Find all of the emissive triangles
        emissive_triangle_ids = self.get_emissive_triangle_indices()
        if len(emissive_triangle_ids) == 0:
            self.has_emissive_triangles = False
        else:
            self.has_emissive_triangles = True
            self.n_emissive_triangles = len(emissive_triangle_ids)
            emissive_triangle_ids = np.array(emissive_triangle_ids, dtype=int)
            self.emissive_triangle_ids = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=int)
            self.emissive_triangle_ids.from_numpy(emissive_triangle_ids)

        # Setup for importance sampling
        if self.has_emissive_triangles:
            # Data Fields
            self.emissive_triangle_areas = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.cdf = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.total_emissive_area = ti.field(shape=(), dtype=float)

            # Compute
            self.compute_emissive_triangle_areas()
            self.compute_cdf()


    def get_emissive_triangle_indices(self) -> List[int]:
        # Iterate over each triangle, and check for emissivity 
        emissive_triangle_ids = []
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            material_id = self.geometry.triangle_material_ids[triangle_id-1]
            emissivity = self.material_library.materials[material_id].Ke
            if emissivity.norm() > 0:
                emissive_triangle_ids.append(triangle_id)

        return emissive_triangle_ids


    @ti.kernel
    def compute_emissive_triangle_areas(self):
        for i in range(self.n_emissive_triangles):
            triangle_id = self.emissive_triangle_ids[i]
            vert_ids = self.geometry.triangle_vertex_ids[triangle_id-1] - 1  # Vertices are indexed from 1
            v0 = self.geometry.vertices[vert_ids[0]]
            v1 = self.geometry.vertices[vert_ids[1]]
            v2 = self.geometry.vertices[vert_ids[2]]

            triangle_area = self.compute_triangle_area(v0, v1, v2)
            self.emissive_triangle_areas[i] = triangle_area
            self.total_emissive_area[None] += triangle_area
        

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        # DONE-xTODO: Compute Area of a triangle given the 3 vertices
        # 
        # Area of a triangle ABC = 0.5 * | AB cross AC |
        # 
        #
        # placholder

        # Compute the vectors representing two sides of the triangle
        AB = v1 - v0
        AC = v2 - v0
        # Compute the cross product of AB and AC
        cross_product = AB.cross(AC)
        # Compute the area of the triangle
        area = 0.5 * cross_product.norm()
        return area


    @ti.kernel
    def compute_cdf(self):
        # DONE-xTODO: Compute the CDF of your emissive triangles
        # self.cdf[i] = ...
        cumulative_sum = 0.0
        ti.loop_config(serialize=True)
        for i in range(self.n_emissive_triangles):
            cumulative_sum += self.emissive_triangle_areas[i]
            self.cdf[i] = cumulative_sum / self.total_emissive_area[None]


    @ti.func
    def sample_emissive_triangle(self) -> int:
        # TODO: Sample an emissive triangle using the CDF
        # return the **index** of the triangle
        #
        # placeholder
        xi = ti.random()
        sampled_index = self.n_emissive_triangles - 1
        ti.loop_config(serialize=True)
        for i in range(self.n_emissive_triangles):
            if xi < self.cdf[i] / self.total_emissive_area[None]:
                sampled_index = i
                break
        return sampled_index


    @ti.func
    def evaluate_probability(self) -> float:
        # DONE-xTODO: return the probabilty of a sample
        #
        # placeholder
        probability_w_i = 1.0 / self.total_emissive_area[None]
        return probability_w_i

    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        sampled_light_triangle_idx = self.sample_emissive_triangle()
        sampled_light_triangle = self.emissive_triangle_ids[sampled_light_triangle_idx]

        # Grab Vertices
        vert_ids = self.geometry.triangle_vertex_ids[sampled_light_triangle-1] - 1  # Vertices are indexed from 1
        
        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]

        # generate point on triangle using random barycentric coordinates
        # https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes#Sampling
        # https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes#SampleUniformTriangle

        # TODO: Sample a direction towards your mesh light
        # given your sampled triangle vertices
        # generate random barycentric coordinates
        # calculate the light direction
        # light direction = (point on light - hit point)
        # don't forget to normalize!
        
        # placeholder
        u0 = ti.random()
        u1 = ti.random()
        if u0 + u1 > 1:
            u0 = 1 - u0
            u1 = 1 - u1

        b0 = u0
        b1 = u1
        b2 = 1 - b0 - b1

        sampled_point = b0 * v0 + b1 * v1 + b2 * v2
        light_direction = tm.normalize(sampled_point - hit_point)
        return light_direction, sampled_light_triangle





@ti.func
def ortho_frames(v_z: tm.vec3) -> tm.mat3:
    random_vec = tm.normalize(tm.vec3([ti.random(), ti.random(), ti.random()]))

    x_axis = tm.cross(v_z, random_vec)
    x_axis = tm.normalize(x_axis)

    y_axis = tm.cross(x_axis, v_z)
    y_axis = tm.normalize(y_axis)

    result = tm.mat3([x_axis, y_axis, v_z]).transpose()

    return result


@ti.func
def reflect(ray_direction:tm.vec3, normal: tm.vec3) -> tm.vec3:
    return (2 * tm.dot(normal, ray_direction) * normal) - ray_direction