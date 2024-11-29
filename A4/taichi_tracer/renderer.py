from enum import IntEnum

import taichi as ti
import taichi.math as tm

from .scene_data import SceneData
from .camera import Camera
from .ray import Ray, HitData
from .sampler import UniformSampler, BRDF, MicrofacetBRDF
from .materials import Material


@ti.data_oriented
class A1Renderer:

    # Enumerate the different shading modes
    class ShadeMode(IntEnum):
        HIT = 1
        TRIANGLE_ID = 2
        DISTANCE = 3
        BARYCENTRIC = 4
        NORMAL = 5
        MATERIAL_ID = 6

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.scene_data = scene_data

        self.shade_mode = ti.field(shape=(), dtype=int)
        self.set_shade_hit()

        # Distance at which the distance shader saturates
        self.max_distance = 10.

        # Numbers used to generate colors for integer index values
        self.r = 3.14159265
        self.b = 2.71828182
        self.g = 6.62607015


    def set_shade_hit(self):          self.shade_mode[None] = self.ShadeMode.HIT
    def set_shade_triangle_ID(self):  self.shade_mode[None] = self.ShadeMode.TRIANGLE_ID
    def set_shade_distance(self):     self.shade_mode[None] = self.ShadeMode.DISTANCE
    def set_shade_barycentrics(self): self.shade_mode[None] = self.ShadeMode.BARYCENTRIC
    def set_shade_normal(self):       self.shade_mode[None] = self.ShadeMode.NORMAL
    def set_shade_material_ID(self):  self.shade_mode[None] = self.ShadeMode.MATERIAL_ID


    @ti.kernel
    def render(self):
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] = color


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        color = tm.vec3(0)
        if   self.shade_mode[None] == int(self.ShadeMode.HIT):         color = self.shade_hit(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.TRIANGLE_ID): color = self.shade_triangle_id(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.DISTANCE):    color = self.shade_distance(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.BARYCENTRIC): color = self.shade_barycentric(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.NORMAL):      color = self.shade_normal(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.MATERIAL_ID): color = self.shade_material_id(hit_data)
        return color
       

    @ti.func
    def shade_hit(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            if not hit_data.is_backfacing:
                color = tm.vec3(1)
            else: 
                color = tm.vec3([0.5,0,0])
        return color


    @ti.func
    def shade_triangle_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            triangle_id = hit_data.triangle_id + 1 # Add 1 so that ID 0 is not black
            r = triangle_id*self.r % 1
            g = triangle_id*self.g % 1
            b = triangle_id*self.b % 1
            color = tm.vec3(r,g,b)
        return color


    @ti.func
    def shade_distance(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            d = tm.clamp(hit_data.distance / self.max_distance, 0,1)
            color = tm.vec3(d)
        return color


    @ti.func
    def shade_barycentric(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            u = hit_data.barycentric_coords[0]
            v = hit_data.barycentric_coords[1]
            w = 1. - u - v
            color = tm.vec3(u,v,w)
        return color


    @ti.func
    def shade_normal(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            normal = hit_data.normal
            color = (normal + 1.) / 2.  # Scale to range [0,1]
        return color


    @ti.func
    def shade_material_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            material_id = hit_data.material_id + 1 # Add 1 so that ID 0 is not black
            r = material_id*self.r % 1
            g = material_id*self.g % 1
            b = material_id*self.b % 1
            color = tm.vec3(r,g,b)
        return color

@ti.data_oriented
class A2Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        MICROFACET = 3

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()


    def set_sample_uniform(self):    self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_brdf(self):       self.sample_mode[None] = self.SampleMode.BRDF
    def set_sample_microfacet(self): self.sample_mode[None] = self.SampleMode.MICROFACET


    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1.0
        for x, y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x, y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / self.iter_counter[None]


    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)

    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        if hit_data.is_hit:
            x = ray.origin + ray.direction * hit_data.distance
            normal = hit_data.normal
            material = self.scene_data.material_library.materials[hit_data.material_id]
            omega_o = -ray.direction
            omega_j = tm.vec3(0)
            pdf = 0.0
            brdf = tm.vec3(0)

            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
                omega_j = UniformSampler.sample_direction()
                pdf = UniformSampler.evaluate_probability()
                brdf_value = BRDF.evaluate_brdf_factor(material, omega_o, omega_j, normal, pdf)
                brdf = brdf_value * max(0.0, tm.dot(normal, omega_j)) / pdf
            elif self.sample_mode[None] == int(self.SampleMode.BRDF):
                omega_j = BRDF.sample_direction(material, omega_o, normal)
                brdf = BRDF.evaluate_brdf(material, omega_o, omega_j, normal)

            elif self.sample_mode[None] == int(self.SampleMode.MICROFACET):
                pass

            shading_point = tm.vec3(x + self.RAY_OFFSET * normal)
            shadow_ray = Ray()
            shadow_ray.origin = shading_point
            shadow_ray.direction = omega_j
            shadow_hit = self.scene_data.ray_intersector.query_ray(shadow_ray)
            shadow_material = self.scene_data.material_library.materials[shadow_hit.material_id]

            # Evaluate the visibility term
            V = 1.0
            if shadow_hit.is_hit and shadow_material.Ke.norm() <= 0.0:
                V = 0.0

            Le = self.scene_data.environment.query_ray(shadow_ray)
            if shadow_hit.is_hit and shadow_material.Ke.norm() > 0:
                Le = shadow_material.Ke

            Lo = Le * V * brdf

            if hit_data.is_hit and material.Ke.norm() > 0:
                color = material.Ke
            elif hit_data.is_hit:
                color = Lo
            else:
                color = self.scene_data.environment.query_ray(ray)

        # else:
        #     color = self.scene_data.environment.query_ray(ray)
        return color


@ti.data_oriented
class EnvISRenderer:
    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        ENVMAP = 2
    
    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.width = width
        self.height = height
        
        self.camera = Camera(width=width, height=height)
        self.count_map = ti.field(dtype=float, shape=(width, height))
        
        self.background = ti.Vector.field(n=3, dtype=float, shape=(width, height))

        self.scene_data = scene_data
        self.sample_mode = ti.field(shape=(), dtype=int)

        self.set_sample_uniform()


    def set_sample_uniform(self): 
        self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_envmap(self):    
        self.sample_mode[None] = self.SampleMode.ENVMAP

    @ti.func
    def render_background(self, x: int, y: int) -> tm.vec3:
        uv_x, uv_y = float(x)/self.width, float(y)/self.height
        uv_x, uv_y = uv_x*self.scene_data.environment.x_resolution, uv_y*self.scene_data.environment.y_resolution
        
        background = self.scene_data.environment.image[int(uv_x), int(uv_y)]
            

        return background


    @ti.kernel
    def render_background(self):
        for x,y in ti.ndrange(self.width, self.height):
            uv_x, uv_y = float(x)/float(self.width), float(y)/float(self.height)
            uv_x, uv_y = uv_x*self.scene_data.environment.x_resolution, uv_y*self.scene_data.environment.y_resolution
            color = self.scene_data.environment.image[int(uv_x), int(uv_y)]

            self.background[x,y] = color

    @ti.kernel
    def sample_env(self, samples: int):
        for _ in ti.ndrange(samples):
            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
                x = int(ti.random() * self.width)
                y = int(ti.random() * self.height)


                self.count_map[x,y] += 1.0
                
            elif self.sample_mode[None] == int(self.SampleMode.ENVMAP):
                sampled_phi_theta = self.scene_data.environment.importance_sample_envmap()
                x = sampled_phi_theta[0] * self.width
                y = sampled_phi_theta[1] * self.height

                self.count_map[int(x), int(y)] += 1.0
    
    @ti.kernel
    def reset(self):
        self.count_map.fill(0.)


@ti.data_oriented
class A3Renderer:
    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        LIGHT = 3
        MIS = 4

    def __init__(
            self,
            width: int,
            height: int,
            scene_data: SceneData
    ) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.canvas_postprocessed = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data
        self.a2_renderer = A2Renderer(width=self.width, height=self.height, scene_data=self.scene_data)

        self.mis_plight = ti.field(dtype=float, shape=())
        self.mis_pbrdf = ti.field(dtype=float, shape=())

        self.mis_plight[None] = 0.5
        self.mis_pbrdf[None] = 0.5

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()
        # self.set_sample_light() # changed for testing
        # self.set_sample_mis() # changed for testing

    def set_sample_uniform(self):
        self.sample_mode[None] = self.SampleMode.UNIFORM
        self.a2_renderer.set_sample_uniform()

    def set_sample_brdf(self):
        self.sample_mode[None] = self.SampleMode.BRDF
        self.a2_renderer.set_sample_brdf()

    def set_sample_light(self):
        self.sample_mode[None] = self.SampleMode.LIGHT

    def set_sample_mis(self):
        self.sample_mode[None] = self.SampleMode.MIS

    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1.0
        for x, y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x, y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / self.iter_counter[None]

    @ti.kernel
    def postprocess(self):
        for x, y in ti.ndrange(self.width, self.height):
            self.canvas_postprocessed[x, y] = tm.pow(self.canvas[x, y], tm.vec3(1.0 / 2.2))
            self.canvas_postprocessed[x, y] = tm.clamp(self.canvas_postprocessed[x, y], xmin=0.0, xmax=1.0)

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)

    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)
        if self.sample_mode[None] == int(self.SampleMode.UNIFORM) or self.sample_mode[None] == int(
                self.SampleMode.BRDF):
            # Uniform or BRDF just calls the A2 renderer
            # DONE-xTODO: Implement Mesh Light support for your A2 renderer
            color = self.a2_renderer.shade_ray(ray)
        else:
            if self.sample_mode[None] == int(self.SampleMode.LIGHT):
                hit_data = self.scene_data.ray_intersector.query_ray(ray)
                x = ray.origin + ray.direction * hit_data.distance
                normal_x = hit_data.normal
                material = self.scene_data.material_library.materials[hit_data.material_id]
                omega_o = -ray.direction

                if hit_data.is_hit and material.Ke.norm() > 0:
                    color = material.Ke
                elif hit_data.is_hit:
                    light_direction, sampled_light_triangle = self.scene_data.mesh_light_sampler.sample_mesh_lights(x)
                    pdf = self.scene_data.mesh_light_sampler.evaluate_probability()
                    brdf = BRDF.evaluate_brdf_factor(material, omega_o, light_direction, normal_x, pdf)

                    y = tm.vec3(x + self.RAY_OFFSET * normal_x)
                    shadow_ray = Ray()
                    shadow_ray.origin = y
                    shadow_ray.direction = light_direction
                    shadow_hit = self.scene_data.ray_intersector.query_ray(shadow_ray)
                    shadow_material = self.scene_data.material_library.materials[shadow_hit.material_id]
                    shadow_normal = shadow_hit.normal

                    max_x = max(0, tm.dot(normal_x, light_direction))
                    max_y = max(0, tm.dot(shadow_normal, -light_direction))

                    Le = tm.vec3(0.)
                    if shadow_hit.is_hit and shadow_material.Ke.norm() > 0:
                        Le = shadow_material.Ke

                    # V = 1.0
                    # if shadow_hit.is_hit and shadow_hit.triangle_id == sampled_light_triangle and shadow_material.Ke.norm() > 0.0:
                    #     V = 0.0

                    Lo = (Le * brdf * max_x * max_y) / (pdf * shadow_hit.distance ** 2)

                    if shadow_hit.is_hit:
                        if shadow_hit.triangle_id == sampled_light_triangle and not hit_data.is_backfacing:
                            color = Lo
                    else:
                        color = 0.0
                else:
                    color = self.scene_data.environment.query_ray(ray)
            if self.sample_mode[None] == int(self.SampleMode.MIS):
                # xTODO: Implement MIS
                hit_data = self.scene_data.ray_intersector.query_ray(ray)
                if hit_data.is_hit:
                    x = ray.origin + ray.direction * hit_data.distance
                    normal_x = hit_data.normal
                    material = self.scene_data.material_library.materials[hit_data.material_id]
                    omega_o = -ray.direction
                    light_direction = tm.vec3(0)
                    sampled_light_triangle = 0
                    brdf = tm.vec3(0)
                    pdf = 0.0
                    if hit_data.is_hit and material.Ke.norm() > 0:
                        color = material.Ke
                    else:
                        u = ti.random()
                        if u < self.mis_plight[None]:
                            # Light sampling
                            light_direction, sampled_light_triangle = self.scene_data.mesh_light_sampler.sample_mesh_lights(
                                x)
                            pdf_light = self.scene_data.mesh_light_sampler.evaluate_probability()
                            pdf = self.mis_plight[None] * pdf_light
                            brdf = BRDF.evaluate_brdf_factor(material, omega_o, light_direction, normal_x, pdf_light)

                            shading_point = tm.vec3(x + self.RAY_OFFSET * normal_x)
                            shadow_ray = Ray()
                            shadow_ray.origin = shading_point
                            shadow_ray.direction = light_direction
                            shadow_hit = self.scene_data.ray_intersector.query_ray(shadow_ray)
                            shadow_material = self.scene_data.material_library.materials[shadow_hit.material_id]
                            shadow_normal = shadow_hit.normal

                            # Light sampling
                            if hit_data.is_hit and material.Ke.norm() > 0:
                                color = material.Ke
                            elif hit_data.is_hit:
                                max_x = max(0, tm.dot(normal_x, light_direction))
                                max_y = max(0, tm.dot(shadow_normal, -light_direction))

                                Le = tm.vec3(0.)
                                if shadow_hit.is_hit and shadow_material.Ke.norm() > 0:
                                    Le = shadow_material.Ke

                                Lo = (Le * brdf * max_x * max_y) / (pdf * shadow_hit.distance ** 2)

                                if shadow_hit.is_hit:
                                    if shadow_hit.triangle_id == sampled_light_triangle and not hit_data.is_backfacing:
                                        color = Lo
                                else:
                                    color = 0.0
                        else:
                            # BRDF sampling
                            light_direction = BRDF.sample_direction(material, omega_o, normal_x)
                            brdf = BRDF.evaluate_brdf(material, omega_o, light_direction, normal_x)

                            shading_point = tm.vec3(x + self.RAY_OFFSET * normal_x)
                            shadow_ray = Ray()
                            shadow_ray.origin = shading_point
                            shadow_ray.direction = light_direction
                            shadow_hit = self.scene_data.ray_intersector.query_ray(shadow_ray)
                            shadow_material = self.scene_data.material_library.materials[shadow_hit.material_id]

                            # Evaluate the visibility term
                            V = 1.0
                            if shadow_hit.is_hit and shadow_material.Ke.norm() <= 0.0:
                                V = 0.0

                            Le = self.scene_data.environment.query_ray(shadow_ray)
                            if shadow_hit.is_hit and shadow_material.Ke.norm() > 0:
                                Le = shadow_material.Ke

                            Lo = Le * V * brdf / self.mis_pbrdf[None]

                            if hit_data.is_hit and material.Ke.norm() > 0:
                                color = material.Ke
                            elif hit_data.is_hit:
                                color = Lo
                            else:
                                color = self.scene_data.environment.query_ray(ray)


                else:
                    color = self.scene_data.environment.query_ray(ray)

        return color


@ti.data_oriented
class A4Renderer:

    # Enumerate the different sampling modes
    class ShadingMode(IntEnum):
        IMPLICIT = 1
        EXPLICIT = 2

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.canvas_postprocessed = ti.Vector.field(n=3, dtype=float, shape=(width, height))

        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data
        
        self.max_bounces = ti.field(dtype=int, shape=())
        # self.max_bounces[None] = 5
        self.max_bounces[None] = 1 # changed for testing

        self.rr_termination_probabilty = ti.field(dtype=float, shape=())
        self.rr_termination_probabilty[None] = 0.0

        self.shading_mode = ti.field(shape=(), dtype=int)
        self.set_shading_implicit()

    def set_shading_implicit(self): self.shading_mode[None] = self.ShadingMode.IMPLICIT
    def set_shading_explicit(self): self.shading_mode[None] = self.ShadingMode.EXPLICIT

    @ti.kernel
    def postprocess(self):
        for x,y in ti.ndrange(self.width, self.height):
            self.canvas_postprocessed[x, y] = tm.pow(self.canvas[x, y], tm.vec3(1.0 / 2.2))
            self.canvas_postprocessed[x, y] = tm.clamp(self.canvas_postprocessed[x, y], xmin=0.0, xmax=1.0)

    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1.0
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] += (color - self.canvas[x,y])/self.iter_counter[None]

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)
        
        if self.shading_mode[None] == int(self.ShadingMode.IMPLICIT):
            color = self.shade_implicit(ray)
        elif self.shading_mode[None] == int(self.ShadingMode.EXPLICIT):
            color = self.shade_explicit(ray)

        return color

    # TODO A4: Implement Implicit Path Tracing
    # xTODO A4: Implement Specular Caustics Support - ECSE 546 Deliverable
    @ti.func
    def shade_implicit(self, ray: Ray) -> tm.vec3:
        throughput = tm.vec3(1.0)
        color = tm.vec3(0.)

        for bounce in range(self.max_bounces[None] + 1):
            hit_data = self.scene_data.ray_intersector.query_ray(ray)
            if not hit_data.is_hit:
                break  # Ray escaped the scene

            x = ray.origin + ray.direction * hit_data.distance
            normal = hit_data.normal
            material = self.scene_data.material_library.materials[hit_data.material_id]
            omega_o = -ray.direction

            if material.Ke.norm() > 0: # Hit an emissive material
                if not hit_data.is_backfacing:
                    color = material.Ke * throughput
                break
            else:
                # Sample a new direction using BRDF importance sampling
                omega_i = BRDF.sample_direction(material, omega_o, normal)
                brdf_factor = BRDF.evaluate_brdf(material, omega_o, omega_i, normal)

                throughput *= brdf_factor

                # Update the ray for the next bounce
                ray.origin = x + self.RAY_OFFSET * normal
                ray.direction = omega_i

        return color

    @ti.func
    def shade_explicit(self, ray: Ray) -> tm.vec3:
        throughput = tm.vec3(1.0)
        color = tm.vec3(0.)

        for bounce in range(self.max_bounces[None]):
            hit_data = self.scene_data.ray_intersector.query_ray(ray)
            if not hit_data.is_hit:
                break  # Ray escaped the scene

            x = ray.origin + ray.direction * hit_data.distance
            normal = hit_data.normal
            material = self.scene_data.material_library.materials[hit_data.material_id]
            omega_o = -ray.direction

            if material.Ke.norm() > 0:  # Hit an emissive material
                if not hit_data.is_backfacing:
                    color = material.Ke * throughput
                break

            # Direct lighting
            light_direction, sampled_light_triangle = self.scene_data.mesh_light_sampler.sample_mesh_lights(x)
            light_pdf = self.scene_data.mesh_light_sampler.evaluate_probability()
            brdf = BRDF.evaluate_brdf(material, omega_o, light_direction, normal)
            brdf_factor = brdf * max(0.0, tm.dot(normal, light_direction)) / (light_pdf)

            shading_point = tm.vec3(x + self.RAY_OFFSET * normal)
            shadow_ray = Ray()
            shadow_ray.origin = shading_point
            shadow_ray.direction = light_direction
            shadow_hit = self.scene_data.ray_intersector.query_ray(shadow_ray)
            if shadow_hit.is_hit and shadow_hit.triangle_id == sampled_light_triangle:
                light_material = self.scene_data.material_library.materials[shadow_hit.material_id]
                if light_material.Ke.norm() > 0:
                    distance = shadow_hit.distance
                    jacobian = max(0.0, tm.dot(shadow_hit.normal, -light_direction)) / (distance * distance)
                    color += light_material.Ke * brdf_factor * jacobian * throughput

            # Russian Roulette termination
            if ti.random() < self.rr_termination_probabilty[None]:
                break

            # Indirect lighting
            omega_i = BRDF.sample_direction(material, omega_o, normal)
            # pdf = BRDF.evaluate_probability(material, omega_o, omega_i, normal) + epsilon
            brdf_factor = BRDF.evaluate_brdf(material, omega_o, omega_i, normal)
            # brdf_factor = brdf * max(0.0, tm.dot(normal, omega_i)) / pdf

            throughput *= brdf_factor / (1.0 - self.rr_termination_probabilty[None])

            # Update the ray for the next bounce
            ray.origin = x + self.RAY_OFFSET * normal
            ray.direction = omega_i

        return color

