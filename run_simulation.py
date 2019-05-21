import os
import metric,snapshot

output_dir = "event_disks_box_movie/"
image_folder = output_dir
video_name = output_dir+'video.avi'

colors = ['r', 'b', 'g', 'orange']

if not os.path.exists(output_dir): os.makedirs(output_dir)

pos = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
vel = [[0.21, 0.12], [0.71, 0.18], [-0.23, -0.79], [0.78, 0.1177]]
singles = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
sigma = 0.15
t = 0.0
n_steps = 1000
save_pic_once_in = 100
next_event, next_event_arg = metric.compute_next_event(pos, vel,sigma,singles,pairs)
snapshot.snapshot(t, pos, vel, colors,sigma,output_dir+'000.png')
for step in range(n_steps):
    next_t = t + next_event
    while t + next_event <= next_t:
        t += next_event
        for k, l in singles: pos[k][l] += vel[k][l] * next_event
        metric.compute_new_velocities(pos, vel, next_event_arg,singles,pairs)
        next_event, next_event_arg = metric.compute_next_event(pos, vel,sigma,singles,pairs)
    remain_t = next_t - t
    for k, l in singles: pos[k][l] += vel[k][l] * remain_t
    t += remain_t
    next_event -= remain_t
    if step%save_pic_once_in==0:
        snapshot.snapshot(t, pos, vel, colors,sigma,output_dir+str(step)+'.png')
        print('time',t)
snapshot.save_video(image_folder,video_name)