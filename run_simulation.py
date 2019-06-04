import os, random, numpy as np
from dump_views import Views
from metric import Metric

n_steps = int(1e3)
save_pic_once_in = 100
epsilon = 1e-4

shape = 'triangle'#'square'#
N_in_row = 20
edge = 1.0 #Not working with edge different from 1 yet
eta = 0.3 #Not working for eta>pi/4~0.78
name = "Melting"

if shape == 'square':
    N = N_in_row ** 2
    pos = [0 for _ in range(N)]

    sigma = np.sqrt((edge ** 2) * eta / (N * np.pi))
    separation_sphere_sphere = (edge+2*sigma)/(N_in_row+1)
    separation_sphere_wall = separation_sphere_sphere-sigma
    for i in range(N_in_row):
        for j in range(N_in_row):
            pos[i*N_in_row+j] = [separation_sphere_wall+float(j)*separation_sphere_sphere,
                                 separation_sphere_wall+float(i)*separation_sphere_sphere]
if shape == 'triangle':
    N_in_col = int(np.floor(2*N_in_row/np.sqrt(3)))
    N = N_in_row * N_in_col
    pos = [0 for _ in range(N)]

    sigma = np.sqrt((edge ** 2) * eta / (N * np.pi))
    separation_sphere_sphere = (edge+2*sigma)/(N_in_row+3/2)
    separation_sphere_wall_big = 3*separation_sphere_sphere/2-sigma
    separation_sphere_wall_small = separation_sphere_sphere - sigma
    dh_wall = (separation_sphere_wall_big+separation_sphere_wall_small)/2
    separation_rows = separation_sphere_sphere*np.sqrt(3)/2
    for i in range(N_in_col):
        for j in range(N_in_row):
            if i % 2 == 0:
                pos[i*N_in_row+j] = [separation_sphere_wall_small+float(j)*separation_sphere_sphere,
                                 dh_wall + float(i)*separation_rows]
            else:
                pos[i*N_in_row + j] = [separation_sphere_wall_big + float(j) * separation_sphere_sphere,
                                         dh_wall + float(i) * separation_rows]

output_dir = name + "_-_"+shape+"_eta=" + str(eta) + "_N=" + str(N)
image_folder = output_dir
video_name = 'video'

if not os.path.exists(output_dir): os.makedirs(output_dir)


free_path = sigma/eta-sigma #approximated, free_path->0 as eta->1 and free_path->inf as eta->0
total_step = N_in_row*free_path

print('Simulation Details:\nN = ' + str(N) + '\nTotal_step = ' + str(total_step))

colors = N*['b']

sim_metric = Metric(sigma, edge)
views_saver = Views(sim_metric, output_dir, colors)

step: int
for step in range(n_steps):
    if step % 4 == 0:
        v_hat = np.array([0, -1])
    if step % 4 == 1:
        v_hat = np.array([1, 0])
    if step % 4 == 2:
        v_hat = np.array([0, 1])
    if step % 4 == 3:
        v_hat = np.array([-1, 0])
    sphere_ind = random.randint(0, len(pos)-1)

    step_left = total_step
    step_counter = 0
    while step_left > 0:
        current_step_size, step_type, sphere_or_wall_ind = \
            sim_metric.step_size(pos, sphere_ind, v_hat, step_left)
        if step % save_pic_once_in == 0 or step == 0:
            step_str = str(step).zfill(int(np.floor(np.log10(n_steps)) + 1)) \
                                 + "." + str(step_counter)
            views_saver.snapshot("step = " + step_str, pos, np.array(v_hat)*step_left/total_step,
                                 sphere_ind, step_str)
            if step_left == total_step: print('step ', step)
        pos[sphere_ind] += current_step_size*v_hat
        if step_type == "wall":
            # define normal vector to the i'th plane, where i=abs(sphere_or_wall_ind)-1
            # as it doesn't matter the sign of n
            n_hat = np.array([0 for _ in pos[0]])
            n_hat[np.abs(sphere_or_wall_ind)-1] = 1
            v_hat = v_hat - 2*np.dot(v_hat, n_hat)*n_hat
        if step_type == "pair_collision":
            sphere_ind = sphere_or_wall_ind
        step_left -= current_step_size
        step_counter += 1
views_saver.save_video(video_name)