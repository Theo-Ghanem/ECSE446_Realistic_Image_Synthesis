import taichi as ti
import taichi.math as tm
import numpy as np

from .ray import Ray


@ti.data_oriented
class Camera:

    def __init__(self, width: int = 128, height: int = 128) -> None:

        # Camera pixel width and height are fixed
        self.width = width
        self.height = height

        # Camera parameters that can be modified are stored as fields
        self.eye = ti.Vector.field(n=3, shape=(), dtype=float)
        self.at = ti.Vector.field(n=3, shape=(), dtype=float)
        self.up = ti.Vector.field(n=3, shape=(), dtype=float)
        self.fov = ti.field(shape=(), dtype=float)

        self.x = ti.Vector.field(n=3, shape=(), dtype=float)
        self.y = ti.Vector.field(n=3, shape=(), dtype=float)
        self.z = ti.Vector.field(n=3, shape=(), dtype=float)

        self.camera_to_world = ti.Matrix.field(n=4, m=4, shape=(), dtype=float)

        # Initialize with some default params
        self.set_camera_parameters(
            eye=tm.vec3([0, 0, 5]),
            at=tm.vec3([0, 0, 0]),
            up=tm.vec3([0, 1, 0]),
            fov=60.
            )


    def set_camera_parameters(
        self, 
        eye: tm.vec3 = None, 
        at: tm.vec3 = None, 
        up: tm.vec3 = None, 
        fov: float = None
        ) -> None:

        if eye: self.eye[None] = eye
        if at: self.at[None] = at
        if up: self.up[None] = up
        if fov: self.fov[None] = fov
        self.compute_matrix()


    @ti.kernel
    def compute_matrix(self):

        '''
        TODO: Copy your A1 solution
        '''


    @ti.func
    def generate_ray(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> Ray:
        
        '''
        TODO: Copy your A1 solution
        '''
        # placeholder
        ray = Ray()
        return ray


    @ti.func
    def generate_ndc_coords(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> tm.vec2:
        
        '''
        TODO: Copy your A1 solution

        '''

        # placeholder
        ndc_x, ndc_y = 0.0, 0.0
        return tm.vec2([ndc_x, ndc_y])

    @ti.func
    def generate_camera_coords(self, ndc_coords: tm.vec2) -> tm.vec4:
        
        '''
        TODO: Copy your A1 solution
        '''

        # palceholder
        cam_x = 0.0
        cam_y = 0.0
        cam_z = 0.0

        return tm.vec4([cam_x, cam_y, cam_z, 0.0])