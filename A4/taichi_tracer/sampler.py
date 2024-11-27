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
        pass


    @staticmethod
    @ti.func
    def evaluate_probability() -> float:
        return 1. / (4. * tm.pi)

#TODO: Implement BRDF Sampling Methods
@ti.data_oriented
class BRDF:
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

    @staticmethod
    @ti.func
    def evaluate_brdf_factor(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass

# Microfacet BRDF based on PBR 4th edition
# https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#
# TODO: Implement Microfacet BRDF Methods
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
            emissive_triangle_ids = np.array(emissive_triangle_ids, dtype=np.int32)
            self.emissive_triangle_ids = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=ti.int32)
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
        # TODO: Compute Area of a triangle given the 3 vertices
        # 
        # Area of a triangle ABC = 0.5 * | AB cross AC |
        # 
        #
        # placholder
        return 1.0


    @ti.kernel
    def compute_cdf(self):
        # TODO: Compute the CDF of your emissive triangles
        # self.cdf[i] = ...
        pass


    @ti.func
    def sample_emissive_triangle(self) -> int:
        # TODO: Sample an emissive triangle using the CDF
        # return the **index** of the triangle
        #
        # placeholder
        return 0

    @ti.func
    def evaluate_probability(self) -> float:
        # TODO: return the probabilty of a sample
        #
        # placeholder
        return 1.0


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
        # generat random barycentric coordinates
        # calculate the light direction
        # light direction = (point on light - hit point)
        # don't forget to normalize!
        
        # placeholder
        light_direction = tm.vec3(1.0)
        return light_direction, sampled_light_triangle



@ti.func
def ortho_frames(v_z: tm.vec3) -> tm.mat3:
    pass


@ti.func
def reflect(ray_direction:tm.vec3, normal: tm.vec3) -> tm.vec3:
    pass