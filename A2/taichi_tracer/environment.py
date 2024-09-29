import taichi as ti
import taichi.math as tm
import numpy as np

from .ray_intersector import Ray


@ti.data_oriented
class Environment:
    def __init__(self, image: np.array):

        self.x_resolution = image.shape[0]
        self.y_resolution = image.shape[1]

        self.image = ti.Vector.field(
            n=3, dtype=float, shape=(self.x_resolution, self.y_resolution)
        )
        self.image.from_numpy(image)

        self.intensity = ti.field(dtype=float, shape=())
        self.set_intensity(1.)


    def set_intensity(self, intensity: float) -> None:
        self.intensity[None] = intensity


    @ti.func
    def query_ray(self, ray: Ray) -> tm.vec3:

        #TODO: Implement equi-rectangular spherical parametrization

        '''
        - compute u and v according to the handout
        - get the x, and y coordinate from the uv coordinates
            - hint: you will need to multiply x and y by their resolution
            - x = u * x_resolution
            - y = v * y_resolution
        - return the environment map query at [x, y]
            - self.image[x, y]
        '''

        pass