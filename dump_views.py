import pylab, cv2, os, numpy as np


class Views:

    def __init__(self, metric, output_dir, colors, arrow_scale=.2):
        self.sigma = metric.sigma
        self.output_dir = output_dir
        self.colors = colors
        self.arrow_scale = arrow_scale

    def snapshot(self, n, pos, vel, sphere_ind, img_name):
        pylab.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
        pylab.gcf().set_size_inches(6, 6)
        pylab.cla()
        pylab.axis([0, 1, 0, 1])
        pylab.setp(pylab.gca(), xticks=[0, 1], yticks=[0, 1])
        for (x, y), c in zip(pos, self.colors):
            circle = pylab.Circle((x, y), radius=self.sigma, fc=c)
            pylab.gca().add_patch(circle)
        (dx, dy) = vel
        dx *= self.arrow_scale
        dy *= self.arrow_scale
        pylab.arrow(pos[sphere_ind][0], pos[sphere_ind][1], dx, dy, fc="k", ec="k",
                    head_width=0.1*np.linalg.norm(vel), head_length=0.1*np.linalg.norm(vel))
        pylab.text(.5, 1.03, 'N = ' + str(n), ha='center')
        pylab.savefig(os.path.join(self.output_dir, img_name+".png"))

    def save_video(self, video_name):
        images = [img for img in os.listdir(self.output_dir) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(self.output_dir, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(os.path.join(self.output_dir,video_name+".avi"),
                                0, 3, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.output_dir, image)))

        cv2.destroyAllWindows()
        video.release()
