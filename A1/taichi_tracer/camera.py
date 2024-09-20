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
        TODO: Compute Camera to World Matrix

        self.camera_to_world[None] = tm.mat4(<You Matrix>)

        '''

        # Compute the camera coordinate frame
        # (orientation of the camera in the world space)
        z_c = (self.at[None] - self.eye[None]).normalized()
        x_c = self.up[None].cross(z_c).normalized() # self.up is up_w in the world space
        y_z = z_c.cross(x_c) # up vector


        # Section responsible for moving camera around
        self.x[None] = x_c
        self.y[None] = y_z
        self.z[None] = z_c

        # Compute the camera to world matrix
        self.camera_to_world[None] = tm.mat4([
            [x_c[0], y_z[0], z_c[0], self.eye[None][0]],
            [x_c[1], y_z[1], z_c[1], self.eye[None][1]],
            [x_c[2], y_z[2], z_c[2], self.eye[None][2]],
            [0, 0, 0, 1]
        ])



    @ti.func
    def generate_ray(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> Ray:
        
        '''
        TODO: Generate Ray

        - generate ndc coords
        - generate camera coods from NDC coords
        - generate a ray
            - ray = Ray()
        - set the ray direction and origin
            - ray.origin = ...
            - ray.direction = ...
        - Ignore jittering for now
        - return ray
        '''

        #Generate ndc coords:
        ndc_coords = self.generate_ndc_coords(pixel_x, pixel_y, jitter)
        # print ("ndc_coords : ", ndc_coords)

        #generate camera coords from NDC coords:
        camera_coords = self.generate_camera_coords(ndc_coords)
        # print ("camera_coords : ", camera_coords)

        ray = Ray()

        # set the ray direction and origin
        ray.origin = self.eye[None]
        ray.direction = (self.camera_to_world[None] @ camera_coords).xyz.normalized()

        return ray


    @ti.func
    def generate_ndc_coords(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> tm.vec2:
        
        '''
        TODO: Generate NDC coods

        - Ignore jittering for now
        
        return tm.vec2([ndc_x, ndc_y])

        '''
        # Convert pixel coordinates to float
        # +0.5 to be in the center of the pixel
        pixel_x_float = ti.cast(pixel_x + 0.5, ti.f32)
        pixel_y_float = ti.cast(pixel_y + 0.5, ti.f32)

        # To convert pixel coordinates to NDC coordinates, we first bring it from [0, width]
        # to [0, 1] and then to [-1, 1] as described in tutorial 1
        ndc_x = (2.0 * pixel_x_float - self.width) / self.width
        ndc_y = (2.0 * pixel_y_float - self.height) / self.height

        return tm.vec2([ndc_x, ndc_y])

    @ti.func
    def generate_camera_coords(self, ndc_coords: tm.vec2) -> tm.vec4:
        
        '''
        TODO: Generate Camera coordinates
        - compute camera_x, camera_y, camera_z
        - return tm.vec4([camera_x, camera_y, camera_z, 0.0])
        '''
        #Generate Camera coordinates
        ndc_x = ndc_coords[0]
        ndc_y = ndc_coords[1]

        aspect_ratio = self.width / self.height

        # Convert the FOV from degres to radians and half it
        half_fov_radians = self.fov[None] * 0.5 * np.pi / 180

        # We then calculate the tangent of the half angle to get the scale factor
        scale = ti.tan(half_fov_radians)

        cam_x = ndc_x * aspect_ratio * scale
        cam_y = ndc_y * scale
        cam_z = 1.0

        return tm.vec4([cam_x, cam_y, cam_z, 0.0]) #homogeneous coordinates 0 for dir vector, 1 for pt vector