import math
import numpy as np

def wall_dist(pos, v_hat,l, sigma,edge):
    """
    :param pos: position vector of the particle
    :param v_hat: direction of next step (norm 1)
    :param l: size of next step
    :param sigma: radius of the sphere/disk
    :param edge: wall location, assuming the walls are of square/cube at
           locations {(0,0,...),(0,edge,...),(edge,edge,...),(edge,0,...),...}
    :return: distance from the closest wall
    """
    edge = edge-sigma #Take care of the sphere radius and forget about it from now on
    v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
    min_dist_to_wall = float('inf')
    for i in range(len(pos)):
        n_hat=[0 for x in pos]
        n_hat[i]=1 #define normal vector to the i'th plane
        print('n_hat='+str(n_hat))
        vn = np.dot(v_hat,n_hat)
        n_dot_p = np.dot(n_hat, pos)
        if not vn ==0:
            dist_to_wall_down = -n_dot_p / vn
            print(dist_to_wall_down)
            if dist_to_wall_down < 0 : dist_to_wall_down = float('inf')
            dist_to_wall_up = (edge - n_dot_p) / vn
            print(dist_to_wall_up)
            if dist_to_wall_up < 0: dist_to_wall_up = float('inf')
            min_dist_to_wall = min(min_dist_to_wall,dist_to_wall_down,dist_to_wall_up)
    if min_dist_to_wall > l: min_dist_to_wall = float('inf')
    return min_dist_to_wall


def pair_dist(pos_a, pos_b, l,v_hat, sigma):
    """
    :param pos_a: position of the particle which is about to make the step
    :param pos_b: position of some other particle
    :param l: size of the step
    :param v_hat: direction of the step
    :param sigma: radius of the sphere
    :return:    the distance particle a needs to travel in order to meet particle b.
                If they don't meet return inf.
    """
    dx = np.array(pos_b)-np.array(pos_a)
    dx_dot_v = np.dot(dx,v_hat)
    discriminant = dx_dot_v**2+4*sigma**2-np.linalg.norm(dx)**2
    if discriminant > 0:
        dist= dx_dot_v - np.sqrt(discriminant)
        if dist < l and dist > 0:
            return dist
    return float('inf')

def compute_next_event(pos, vel,sigma,singles,pairs):
    wall_times = [wall_time(pos[k][l], vel[k][l], sigma) for k, l in singles]
    pair_times = [pair_time(pos[k], vel[k], pos[l], vel[l], sigma) for k, l in pairs]
    l=wall_times + pair_times
    return min(zip(l, range(len(l))))


def compute_new_velocities(pos, vel, next_event_arg,singles,pairs):
    if next_event_arg < len(singles):
        collision_disk, direction = singles[next_event_arg]
        vel[collision_disk][direction] *= -1.0
    else:
        a, b = pairs[next_event_arg - len(singles)]
        del_x = [pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]]
        abs_x = math.sqrt(del_x[0] ** 2 + del_x[1] ** 2)
        e_perp = [c / abs_x for c in del_x]
        del_v = [vel[b][0] - vel[a][0], vel[b][1] - vel[a][1]]
        scal = del_v[0] * e_perp[0] + del_v[1] * e_perp[1]
        for k in range(2):
            vel[a][k] += e_perp[k] * scal
            vel[b][k] -= e_perp[k] * scal

