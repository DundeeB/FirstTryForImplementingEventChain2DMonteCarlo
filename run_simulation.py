import os, random, numpy as np
import metric,snapshot

output_dir = "event_disks_box_movie/"
image_folder = output_dir
video_name = output_dir+'video.avi'

if not os.path.exists(output_dir): os.makedirs(output_dir)

pos = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
colors = ['r','b','y','m']
sigma = 0.05
total_step = 0.4
edge = 1
n_steps = 100
save_pic_once_in = 1
epsilon = 1e-4
snapshot.snapshot(0, pos, [1,0], 0, colors,sigma,output_dir+'000.png')
for step in range(n_steps):
    if step % 2 == 0:
        v_hat = np.array([0, -1])
    else:
        v_hat = np.array([1, 0])
    sphere_ind = random.randint(0, len(pos)-1)

    if step % save_pic_once_in == 0:
        snapshot.snapshot(step, pos, v_hat, sphere_ind, \
                          colors,sigma,output_dir+str(step)+'.png')
        print('step ',step)

    step_left = total_step
    while step_left > 0:
        current_step_size, step_type, sphere_or_wall_ind = \
            metric.step_size(pos, sphere_ind, sigma, v_hat, total_step, edge)
        pos[sphere_ind] += current_step_size*v_hat
        if step_type == "wall":
            # define normal vector to the i'th plane, where i=abs(sphere_or_wall_ind)
            # as it doesn't matter the sign of n
            n_hat = np.array([0 for x in pos[0]])
            n_hat[np.abs(sphere_or_wall_ind)] = 1
            v_hat = v_hat - 2*np.dot(v_hat, n_hat)*n_hat
        if step_type == "pair_collision":
            sphere_ind = sphere_or_wall_ind
        step_left -= current_step_size
        if current_step_size<epsilon:
            print('current_step_size<epsilon!')
            break
snapshot.save_video(image_folder, video_name)