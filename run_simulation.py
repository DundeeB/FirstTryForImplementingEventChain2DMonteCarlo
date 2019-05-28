import os, random, numpy as np
from dump_views import Views
from metric import Metric

output_dir = "event_disks_box_movie"
image_folder = output_dir
video_name = 'video'

if not os.path.exists(output_dir): os.makedirs(output_dir)

N = 10
pos = [0 for _ in range(N**2)]
for i in range(N):
    for j in range(N):
        pos[i*N+j] = [float(i+1)/(N+1), float(j+1)/(N+1)]
#pos = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
colors = (N**2)*['r']
sigma = 0.025
total_step = 0.4
edge = 1
n_steps = 100
save_pic_once_in = 1
epsilon = 1e-4
sim_metric = Metric(sigma, edge)
views_saver = Views(sim_metric, output_dir, colors)

step: int
for step in range(n_steps):
    if step % 2 == 0:
        v_hat = np.array([0, -1])
    else:
        v_hat = np.array([1, 0])
    sphere_ind = random.randint(0, len(pos)-1)

    step_left = total_step
    step_counter=0
    while step_left > 0:
        current_step_size, step_type, sphere_or_wall_ind = \
            sim_metric.step_size(pos, sphere_ind, v_hat, step_left)
        if step % save_pic_once_in == 0 or step == 0:
            views_saver.snapshot(step, pos, np.array(v_hat)*step_left,
                                 sphere_ind, str(step).zfill(int(np.floor(np.log10(n_steps)) + 1))
                                 + "." + str(step_counter))
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