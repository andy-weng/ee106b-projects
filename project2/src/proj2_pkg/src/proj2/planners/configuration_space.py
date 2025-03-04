#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena

Updated to:
- Provide an adaptive_local_plan() method
- Increase sub-stepping in check_path_collision()
"""

import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

class Plan(object):
    def __init__(self, times, target_positions, open_loop_inputs, dt=0.01):
        self.dt = dt
        self.times = times
        self.positions = target_positions
        self.open_loop_inputs = open_loop_inputs

    def __iter__(self):
        for t, p, c in zip(self.times, self.positions, self.open_loop_inputs):
            yield t, p, c

    def __len__(self):
        return len(self.times)

    def get(self, t):
        index = int(np.sum(self.times <= t))
        index = index - 1 if index else 0
        return self.positions[index], self.open_loop_inputs[index]

    def end_position(self):
        return self.positions[-1]

    def start_position(self):
        return self.positions[0]

    def get_prefix(self, until_time):
        times = self.times[self.times <= until_time]
        positions = self.positions[self.times <= until_time]
        inputs = self.open_loop_inputs[self.times <= until_time]
        return Plan(times, positions, inputs, dt=self.dt)

    @classmethod
    def chain_paths(cls, *paths):
        def chain_two_paths(path1, path2):
            if not path1 and not path2:
                return None
            elif not path1:
                return path2
            elif not path2:
                return path1
            assert path1.dt == path2.dt, "Different dt."
            assert np.allclose(path1.end_position(), path2.start_position()), (
                "Inconsistent start/end positions."
            )
            times = np.concatenate((path1.times, path1.times[-1] + path2.times[1:]), axis=0)
            positions = np.concatenate((path1.positions, path2.positions[1:]), axis=0)
            inputs = np.concatenate((path1.open_loop_inputs, path2.open_loop_inputs[1:]), axis=0)
            return Plan(times, positions, inputs, dt=path1.dt)

        chained = None
        for p in paths:
            chained = chain_two_paths(chained, p)
        return chained

@contextmanager
def expanded_obstacles(obstacle_list, delta):
    for obs in obstacle_list:
        obs[2] += delta
    yield obstacle_list
    for obs in obstacle_list:
        obs[2] -= delta

class ConfigurationSpace(object):
    def __init__(self, dim, low_lims, high_lims, obstacles, dt=0.01):
        self.dim = dim
        self.low_lims = np.array(low_lims)
        self.high_lims = np.array(high_lims)
        self.obstacles = obstacles
        self.dt = dt

    def distance(self, c1, c2):
        pass

    def sample_config(self, *args):
        pass

    def check_collision(self, c):
        pass

    def check_path_collision(self, path):
        pass

    def local_plan(self, c1, c2, dt=0.01):
        pass

    def nearest_config_to(self, config_list, config):
        config_list = [np.array(c) for c in config_list]
        config = np.array(config)
        return min(config_list, key=lambda c: self.distance(c, config))
    
class FreeEuclideanSpace(ConfigurationSpace):
    def __init__(self, dim, low_lims, high_lims, sec_per_meter=4):
        super(FreeEuclideanSpace, self).__init__(dim, low_lims, high_lims, [])
        self.sec_per_meter = sec_per_meter

    def distance(self, c1, c2):
        return np.linalg.norm(c1 - c2)

    def sample_config(self, *args):
        return np.random.uniform(self.low_lims, self.high_lims).reshape((self.dim,))

    def check_collision(self, c):
        return False

    def check_path_collision(self, path):
        return False

    def local_plan(self, c1, c2):
        v = c2 - c1
        dist = np.linalg.norm(c1 - c2)
        total_time = dist * self.sec_per_meter
        vel = v / total_time
        p = lambda t: (1 - (t / total_time)) * c1 + (t / total_time) * c2
        times = np.arange(0, total_time, self.dt)
        positions = p(times[:, None])
        velocities = np.tile(vel, (positions.shape[0], 1))
        plan = Plan(times, positions, velocities, dt=self.dt)
        return plan

class BicycleConfigurationSpace(ConfigurationSpace):
    def __init__(self, low_lims, high_lims, input_low_lims, input_high_lims, obstacles, robot_radius):
        super().__init__(4, low_lims, high_lims, obstacles)
        self.robot_radius = robot_radius
        self.robot_length = 0.3
        self.input_low_lims = input_low_lims
        self.input_high_lims = input_high_lims

    def distance(self, c1, c2):
        c1, c2 = np.array(c1), np.array(c2)
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        dist_xy = np.sqrt(dx**2 + dy**2)

        def angle_diff(a, b):
            d = (a - b) % (2*np.pi)
            if d > np.pi:
                d -= 2*np.pi
            return abs(d)

        dtheta = angle_diff(c1[2], c2[2])
        dphi   = abs(c2[3] - c1[3])
        return dist_xy + 0.2*dtheta + 0.2*dphi

    def sample_config(self, *args):
        goal = args[0] if args else None
        p_goal = 0.2
        for _ in range(100):
            if goal is not None and np.random.rand() < p_goal:
                rand_config = np.array(goal)
            else:
                rand_config = np.random.uniform(self.low_lims, self.high_lims)
            if not self.check_collision(rand_config):
                return rand_config
        print("Warning: No valid sample found after 100 tries.")
        return np.random.uniform(self.low_lims, self.high_lims)

    def check_collision(self, c):
        x, y, theta, phi = c
        if not (self.low_lims[0] <= x <= self.high_lims[0]):
            return True
        if not (self.low_lims[1] <= y <= self.high_lims[1]):
            return True
        if not (self.low_lims[2] <= theta <= self.high_lims[2]):
            return True
        if not (self.low_lims[3] <= phi <= self.high_lims[3]):
            return True
        for (ox, oy, r) in self.obstacles:
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < (r + self.robot_radius):
                return True
        return False

    def check_path_collision(self, path):
        if path is None:
            return True
        sub_steps = 5
        for i in range(len(path.positions) - 1):
            start_pos = path.positions[i]
            end_pos   = path.positions[i+1]
            u1, u2 = path.open_loop_inputs[i]
            if not (self.input_low_lims[0] <= u1 <= self.input_high_lims[0]):
                return True
            if not (self.input_low_lims[1] <= u2 <= self.input_high_lims[1]):
                return True
            for alpha in np.linspace(0, 1, sub_steps + 1):
                inter_pos = (1 - alpha)*start_pos + alpha*end_pos
                if self.check_collision(inter_pos):
                    return True
        if len(path.positions) > 0:
            if self.check_collision(path.positions[-1]):
                return True
        return False

    def local_plan(self, c1, c2, dt=0.01):
        return self._local_plan_internal(c1, c2, local_T=0.15, dt=dt)

    def adaptive_local_plan(self, c1, c2, dt=0.01):
        return self._local_plan_internal(c1, c2, local_T=0.075, dt=dt)

    def local_plan(self, c1, c2, dt=0.01):
        c1 = np.array(c1)
        c2 = np.array(c2)
        local_T = 0.5  
        max_steps = int(local_T / dt)
        if np.linalg.norm(c1[:2] - c2[:2]) < 0.05: 
            print("Performing Point Turn!")
            velocities = [
                (0.0, self.input_high_lims[1]),
                (0.0, -self.input_high_lims[1]) 
            ]
        else:
            velocities = [
                (self.input_high_lims[0], 0.0),
                (-self.input_high_lims[0], 0.0),
                (self.input_high_lims[0], self.input_high_lims[1]),
                (self.input_high_lims[0], -self.input_high_lims[1]),
                (-self.input_high_lims[0], self.input_high_lims[1]),
                (-self.input_high_lims[0], -self.input_high_lims[1])
            ]
        best_plan = None
        best_dist = float('inf')
        for (u1, u2) in velocities:
            path_positions = []
            path_inputs = []
            times = []
            state = c1.copy()
            t = 0.0
            for _ in range(max_steps):
                path_positions.append(state.copy())
                path_inputs.append([u1, u2])
                times.append(t)
                x, y, th, phi = state
                x_next = x + np.cos(th) * u1 * dt
                y_next = y + np.sin(th) * u1 * dt
                th_next = th + (np.tan(phi) / self.robot_length) * u1 * dt
                phi_next = phi + u2 * dt
                phi_next = np.clip(phi_next, self.low_lims[3], self.high_lims[3])
                state = np.array([x_next, y_next, th_next, phi_next], dtype=float)
                t += dt
                if t > local_T:
                    break
            plan_candidate = Plan(np.array(times), np.array(path_positions), np.array(path_inputs), dt)
            final_dist = self.distance(plan_candidate.end_position(), c2)
            if final_dist < best_dist:
                best_dist = final_dist
                best_plan = plan_candidate
        return best_plan
