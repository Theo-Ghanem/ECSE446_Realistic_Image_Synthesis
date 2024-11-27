import taichi as ti
import taichi.math as tm
import numpy as np

from .ray_intersector import Ray


@ti.data_oriented
class Environment:
    def __init__(self, image: np.array):

        self.x_resolution = image.shape[0]
        self.y_resolution = image.shape[1]

        # original env map
        self.image = ti.Vector.field(
            n=3, dtype=float, shape=(self.x_resolution, self.y_resolution)
        )

        # luminance env map             
        # luminance = 0.2126*rgb.x + 0.7152*rgb.y + 0.0722*rgb.z
        self.image_scalar = ti.field(
            dtype=float, shape=(self.x_resolution, self.y_resolution)
        )

        # p(theta) marginal
        self.marginal_ptheta = ti.field(
            dtype=float, shape=(self.y_resolution)
        )

        # cdf of p(theta)
        self.cdf_ptheta = ti.field(
            dtype=float, shape=(self.y_resolution)
        )

        # p(phi | theta)
        self.conditional_p_phi_given_theta = ti.field(
            dtype=float, shape=(self.x_resolution, self.y_resolution)
        )

        # cdf of p(phi | theta)
        self.cdf_p_phi_given_theta = ti.field(
            dtype=float, shape=(self.x_resolution, self.y_resolution)
        )

        self.image.from_numpy(image)

        self.intensity = ti.field(dtype=float, shape=())
        self.set_intensity(1.)


    def set_intensity(self, intensity: float) -> None:
        self.intensity[None] = intensity


    @ti.func
    def query_ray(self, ray: Ray) -> tm.vec3:
        #TODO: copy your A2 solution
        pass

    @ti.kernel
    def precompute_envmap(self):
        self.precompute_scalar()
        self.precompute_marginal_ptheta()
        self.precompute_conditional_p_phi_given_theta()
        self.precompute_cdfs()


    @ti.func
    def precompute_scalar(self):
        # TODO: 546 Deliverable Only
        # Precompute the scalar version of your environment map
        # this is your join distribution p(phi, theta)
        #  self.image_scalar[x, y] = luminance*sin_theta

        pass

    @ti.func
    def precompute_marginal_ptheta(self):
        # TODO: 546 Deliverable Only
        # Precompute the marginal distribution p(theta)
        # 1 - compute the cumulative sum of the row for each entry of theta
        # 2 - normalize the marginal dsitribution
        # self.marginal_ptheta[y] = ...
        pass

    @ti.func
    def precompute_conditional_p_phi_given_theta(self):
        # TODO: 546 Deliverable Only
        # Compute conditional distribution p(phi | theta)
        # p (phi | theta) = p(phi, theta) / p(theta)
        # self.conditional_p_phi_given_theta[x, y]
        pass

    @ti.func
    def precompute_cdfs(self):
        # TODO: 546 Deliverable Only
        # Compute the CDF for p(theta) and p(phi | theta)
        # self.cdf_ptheta
        pass

    @ti.func
    def sample_theta(self, u1: float) -> int:
        # TODO: 546 Deliverable Only
        # given a uniform random value, return the corresponding **index** of theta
        # 
        # ==== THIS IS JUST AN EXAMPLE OF THE RETURN TYPE ===
        # === THIS IS NOT THE SOLUTION ===
        #
        # if self.cdf_ptheta[i] < u1:
        #    return i
        #
        # placeholder
        return 0

    @ti.func
    def sample_phi(self, theta: int, u2: float) -> int: 
        # TODO: 546 Deliverable Only
        # given a uniform random value, and theta, return the corresponding **index** of phi
        # 
        # ==== THIS IS JUST AN EXAMPLE OF THE RETURN TYPE ===
        # === THIS IS NOT THE SOLUTION ===
        #
        # if self.cdf_p_phi_given_theta[i, theta] < u2:
        #   return i
        #
        # placeholder
        return 0
    

    @ti.func
    def importance_sample_envmap(self) -> tm.vec2:
        
        u1 = ti.random()
        u2 = ti.random()

        sampled_theta = self.sample_theta(u1)
        sampled_phi = self.sample_phi(sampled_theta, u2)
        
        # TODO: 546 Deliverable Only

        # Once you have found the sampled theta and sampled phi indices
        # You will need to interpolate them to their actual value
        
        # Ex: if your theta cdf is [0.0, 0.5, 0.1]
        # for values theta         [1.0, 2.0, 3.0]

        # if u1 = 0.25
        # sampled_phi = 1.0
        # lerped_phi should be 1.5
        # since 0.25 is halfway between 0.0 and 0.5
        
        # Finally, return the normalized coordinates u and v 
        # u = (lerped_phi   / float(self.x_resolution))
        # v = (lerped_theta / float(self.y_resolution))

        # return tm.vec2([u, v])
        
        
        
        # palceholder
        return tm.vec2([0., 0.])


@ti.func
def lerp(x: float, a: float, b: float) -> float:
    return ((1.0-x) * a) + (x * b)