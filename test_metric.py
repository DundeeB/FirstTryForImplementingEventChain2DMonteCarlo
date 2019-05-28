from unittest import TestCase
from metric import Metric
import numpy as np

class TestMetric(TestCase):

    def test_wall_dist(self):
        test_metric = Metric(sigma=0.1, edge=1)
        pos = [0.5, 0.5]
        for wall in [1,-1,2,-2]:
            v_hat = [0,0]
            v_hat[np.abs(wall)-1] = np.sign(wall)
            dist_to_wall, wall_type = test_metric.wall_dist(pos, v_hat, 1)
            self.assertEqual(dist_to_wall, 0.4, "Wrong distance to wall")
            self.assertEqual(wall_type, wall, "Wrong Classification of wall")

    def test_wall_dist_starting_at_the_wall(self):
        test_metric = Metric(sigma=0.1, edge=1)
        pos = [0.1, 0.5]

        dist_to_wall, wall_type = test_metric.wall_dist(pos, [-1,0], 1)
        self.assertEqual(dist_to_wall, 0, "Wrong distance to wall")
        self.assertEqual(wall_type, -1, "Wrong Classification of wall")

        dist_to_wall, wall_type = test_metric.wall_dist(pos, [0, 1], 1)
        self.assertEqual(dist_to_wall, 0.4, "Wrong distance to wall")
        self.assertEqual(wall_type, 2, "Wrong Classification of wall")

    def test_pair_dist(self):
        sigma = 0.1
        test_metric = Metric(sigma, edge=1)
        positions = [[0.2, 0.2], [0.7, 0.2]]
        self.assertAlmostEqual(test_metric.pair_dist(positions[0], positions[1], 10, [1, 0]),
                         0.5-2*sigma, 1, "Wrong distance between pairs")

        positions = [[0.2, 0.2], [0.7, 0.7]]
        self.assertAlmostEqual(test_metric.pair_dist(positions[0], positions[1], 10, [0, 1]),
                               float('inf'), 1, "Should not collide")

    def test_pair_dist_starting_at_zero_dist(self):
        sigma = 0.1
        test_metric = Metric(sigma, edge=1)
        positions = [[0.2, 0.2], [0.2+2*sigma, 0.2]]
        self.assertAlmostEqual(test_metric.pair_dist(positions[0], positions[1], 10, [1, 0]),
                               0, 1, "Should collide")

    def test_step_size(self):
        sigma = 0.1
        test_metric = Metric(sigma, edge=1)
        positions = [[0.2, 0.2], [0.7, 0.2]]
        current_step_size, step_type, sphere_or_wall_ind = \
            test_metric.step_size(positions, 0, [1, 1], .2)
        self.assertEqual(current_step_size, .2, "Problem with free step")
        current_step_size, step_type, sphere_or_wall_ind = \
            test_metric.step_size(positions, 0, [1, 0], 1)
        self.assertAlmostEqual(current_step_size, .3, 1, "Problem with pair collision")


    def test_step_size_start_zero_dist(self):
        sigma = 0.1
        test_metric = Metric(sigma, edge=1)
        positions = [[0.2, 0.2], [0.4, 0.2]]

        current_step_size, step_type, sphere_or_wall_ind = \
            test_metric.step_size(positions, 0, [1, 0], 1)
        self.assertEqual(current_step_size, 0, "Problem with collision distance")
        self.assertEqual(step_type, "pair_collision", "Problem with collision")

        current_step_size, step_type, sphere_or_wall_ind = \
            test_metric.step_size(positions, 0, [0, 1], .2)
        self.assertEqual(current_step_size, .2, "Problem should not collide")
        self.assertEqual(step_type, "Hits_nothing", "Problem shouldn't hit anything")

        current_step_size, step_type, sphere_or_wall_ind = \
            test_metric.step_size(positions, 0, [0, -1], .2)
        self.assertEqual(current_step_size, .1, "Problem should hit the wall")
        self.assertEqual(step_type, "wall", "Problem shouldn't hit anything")

        x = np.sqrt(sigma**2-0.05**2)
        positions = [[0.2, 0.2], [0.4+x, 0.15]]
        current_step_size, step_type, sphere_or_wall_ind = \
            test_metric.step_size(positions, 0, [1, 0], 1)
        self.assertEqual(step_type, "pair_collision", "Problem with collision")
        self.assertAlmostEqual(current_step_size, 0.1, 1, "Problem with collision distance")

        positions = [[0.2, 0.2], [0.2+0.1*np.sqrt(2), 0.2+0.1*np.sqrt(2)]]
        current_step_size, step_type, sphere_or_wall_ind = \
            test_metric.step_size(positions, 0, [1, 0], 1)
        self.assertEqual(step_type, "pair_collision", "Problem with collision")
        self.assertAlmostEqual(current_step_size, 0, 1, "Problem with collision distance")