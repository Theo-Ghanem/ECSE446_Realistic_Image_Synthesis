from abc import ABC, abstractmethod

import taichi as ti
import taichi.math as tm

from .geometry import Geometry
from .materials import Material
from .ray import Ray, HitData


@ti.data_oriented
class RayIntersector(ABC):

    def __init__(self, geometry: Geometry):
        self.EPSILON = 1e-7
        self.geometry = geometry


    @abstractmethod
    @ti.func
    def query_ray(ray: Ray) -> HitData:
        pass


    @ti.func
    def intersect_triangle(self, ray: Ray, triangle_id: int) -> HitData:

        hit_data = HitData()

        # Grab Vertices
        vert_ids = self.geometry.triangle_vertex_ids[triangle_id-1] - 1  # Vertices are indexed from 1
        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]

        # Normals at each vertex
        normal_indices = self.geometry.triangle_normal_ids[triangle_id-1]-1

        normal_0 = self.geometry.normals[normal_indices[0]]
        normal_1 = self.geometry.normals[normal_indices[1]]
        normal_2 = self.geometry.normals[normal_indices[2]]

        # Material of the triangle
        material_id = self.geometry.triangle_material_ids[triangle_id-1]


        '''
        TODO: Ray-Triangle intersection
        
        - v0, v1, v2 are the vertices of a given triangle
        - normal_0, normal_1, normal2 are the normals at each vertex

        # check if the ray intersects inside the given triangle
        # if it is, fill the hit_data with the following information
        '''

        #1st compute edge vectors:
        e1 = v1 - v0
        e2 = v2 - v0

        det = tm.cross(ray.direction, e2).dot(e1)

        if -self.EPSILON < det < self.EPSILON: # if det close to 0 -> ray is parallel to the triangle
            hit_data.is_hit = False
        else: # it intersects the triangle, inside or outside?
            #compute the barycentric coordinates:
            u = (ray.origin - v0).dot(tm.cross(ray.direction, e2))/det
            v = (ray.direction.dot(tm.cross((ray.origin-v0), e1))) / det
            w = 1.0 - u - v

            # We then verify whether the barycentric coordinates fall within the sheared triangle frame bounds:
            if 0 <= u <= 1 and 0 <= v <= 1 and u+v <= 1:
                # If all these tests pass, then the ray intersects the triangle at parametric distance t:
                t = (e2.dot(tm.cross((ray.origin - v0), e1)))/det

                #To account for numerics, we only consider intersections for 𝑡 > EPSILON
                if t < self.EPSILON: #check if the intersection is behind the ray
                    hit_data.is_hit = False

                #check if the intersection is closer than the current closest intersection

                else: # We compute the intersection point:
                    hit_data.is_hit = True
                    hit_data.is_backfacing = det < 0
                    hit_data.triangle_id = triangle_id
                    hit_data.distance = (ray.origin - v0).dot(tm.cross(e1, e2)) / det
                    hit_data.barycentric_coords = tm.vec2(u, v)
                    #the per-pixel interpolated normal, returned as a taichi.math vec3 type (hint: you will need to flip your normal if your triangle is backfacing, as well as performing the per-vertex to per-pixel interpolation and renormalization)
                    if hit_data.is_backfacing:
                        hit_data.normal = w*normal_0 + u*normal_1 + v*normal_2
                    else:
                        hit_data.normal = w*normal_0 + v*normal_1 + u*normal_2
                    hit_data.material_id = material_id
        return hit_data


@ti.data_oriented
class BruteForceRayIntersector(RayIntersector):

    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)


    @ti.func
    def query_ray(self, ray: Ray) -> HitData:

        closest_hit = HitData()
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            hit_data = self.intersect_triangle(ray, triangle_id)

            if hit_data.is_hit:
                if (hit_data.distance < closest_hit.distance) or (not closest_hit.is_hit):
                    closest_hit = hit_data

        return closest_hit