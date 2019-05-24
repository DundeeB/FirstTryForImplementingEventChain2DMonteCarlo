import pylab
import cv2
import os

def snapshot(n, pos, vel, sphere_ind, colors,sigma,output_pwd, arrow_scale=.2):
    pylab.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
    pylab.gcf().set_size_inches(6, 6)
    pylab.cla()
    pylab.axis([0, 1, 0, 1])
    pylab.setp(pylab.gca(), xticks=[0, 1], yticks=[0, 1])
    for (x, y), c in zip(pos, colors):
        circle = pylab.Circle((x, y), radius=sigma, fc=c)
        pylab.gca().add_patch(circle)
    (dx, dy) = vel
    dx *= arrow_scale
    dy *= arrow_scale
    pylab.arrow(pos[sphere_ind][0], pos[sphere_ind][1], dx, dy, fc="k", ec="k", head_width=0.05, head_length=0.05)
    pylab.text(.5, 1.03, 'N = ' + str(n), ha='center')
    pylab.savefig(output_pwd)

def save_video(image_folder,video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))


    cv2.destroyAllWindows()
    video.release()