import numpy as np

epsilon = 1e-4


class Metric:

    def __init__(self, sigma, edge):
        self.sigma = sigma
        self.edge = edge

    def wall_dist(self, pos, v_hat, l):
        """
        :param pos: position vector of the particle
        :param v_hat: direction of next step (norm 1)
        :param l: size of next step
        :return: distance from the closest wall
        """
        v_hat = np.array(v_hat) / np.linalg.norm(v_hat)
        min_dist_to_wall = float('inf')
        wall_type = np.nan
        for i in range(len(pos)):
            n_hat = [0 for x in pos]
            n_hat[i] = 1  # define normal vector to the i'th plane
            vn = np.dot(v_hat, n_hat)
            n_dot_p = np.dot(n_hat, pos)
            if not vn == 0:
                dist_to_wall_down = -n_dot_p / vn-np.abs(self.sigma/vn)
                if dist_to_wall_down < 0: dist_to_wall_down = float('inf')
                if dist_to_wall_down < min_dist_to_wall:
                    wall_type = -i-1
                    min_dist_to_wall = dist_to_wall_down
                dist_to_wall_up = (self.edge - n_dot_p) / vn-np.abs(self.sigma/vn)
                if dist_to_wall_up < 0: dist_to_wall_up = float('inf')
                if dist_to_wall_up < min_dist_to_wall:
                    wall_type = i+1
                    min_dist_to_wall = dist_to_wall_up
        if min_dist_to_wall > l:
            min_dist_to_wall = float('inf')
            wall_type = np.nan
        return min_dist_to_wall, wall_type

    def pair_dist(self, pos_a, pos_b, l, v_hat):
        """
        :param pos_a: position of the particle which is about to make the step
        :param pos_b: position of some other particle
        :param l: size of the step
        :param v_hat: direction of the step
        :param sigma: radius of the sphere
        :return:    the distance particle a needs to travel in order to meet particle b.
                    If they don't meet return inf.
        """
        v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
        dx = np.array(pos_b) - np.array(pos_a)
        dx_dot_v = np.dot(dx, v_hat)
        if dx_dot_v <= 0:
            return float('inf')
        if np.linalg.norm(dx) - 2*self.sigma <= epsilon:
            return 0
        discriminant = dx_dot_v ** 2 + 4 * self.sigma ** 2 - np.linalg.norm(dx) ** 2
        if discriminant > 0:
            dist: float = dx_dot_v - np.sqrt(discriminant)
            if dist <= l and dist >= 0: return dist
        return float('inf')

    def step_size(self, positions, sphere_ind, v_hat, l):
        closest_wall, wall_type = self.wall_dist(positions[sphere_ind], v_hat, l)
        spheres_dists = [float('inf') for _ in range(len(positions))]
        for i in range(len(positions)):
            if i != sphere_ind:
                spheres_dists[i] = self.pair_dist(positions[sphere_ind], positions[i], l, v_hat)
        other_sphere = np.argmin(spheres_dists)
        dist_sphere = spheres_dists[other_sphere]
        if closest_wall < dist_sphere:  # it hits a wall
            return closest_wall, "wall", wall_type
        if closest_wall > dist_sphere:  # it hits another sphere
            return dist_sphere, "pair_collision", other_sphere
        return l, "Hits_nothing", np.nan
