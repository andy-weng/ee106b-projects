#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena

Updated to:
- Use an adaptive local planning approach with smaller dt
- Use a shorter prefix_time_length
- Attempt multiple expansions if collisions occur
"""

import sys
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .configuration_space import BicycleConfigurationSpace, Plan

class RRTGraph(object):
    def __init__(self, *nodes):
        """
        Basic graph structure to store RRT nodes and edges.
        """
        self.nodes = [n for n in nodes]
        self.parent = defaultdict(lambda: None)
        self.path = defaultdict(lambda: None)

    def add_node(self, new_config, parent, path):
        """
        Add a new configuration to the graph, along with
        an edge (path) from parent to new_config.
        """
        new_config = tuple(new_config)
        parent = tuple(parent)
        self.nodes.append(new_config)
        self.parent[new_config] = parent
        self.path[(parent, new_config)] = path

    def get_edge_paths(self):
        """
        Yields all edges (paths) in the current RRT graph.
        """
        for pair in self.path:
            yield self.path[pair]

    def construct_path_to(self, c):
        """
        Recursively builds the path from the start node to c.
        """
        c = tuple(c)
        if self.parent[c] is None:
            return None
        return Plan.chain_paths(
            self.construct_path_to(self.parent[c]),
            self.path[(self.parent[c], c)]
        )

class RRTPlanner(object):
    def __init__(self, config_space, max_iter=15000, expand_dist=0.3):
        """
        config_space: an instance of BicycleConfigurationSpace
        max_iter: maximum RRT iterations
        expand_dist: distance threshold to goal for stopping
        """
        self.config_space = config_space
        self.max_iter = max_iter
        self.expand_dist = expand_dist
        self.graph = None
        self.plan = None

    def plan_to_pose(self, start, goal, dt=0.005, prefix_time_length=0.2):
        """
        Uses the RRT algorithm to plan from 'start' to 'goal'.
        dt=0.005 => smaller time step for finer expansions
        prefix_time_length=0.2 => short expansions
        """
        print("======= Planning with RRT (Adaptive Collisions) =======")
        self.graph = RRTGraph(start)
        self.plan = None

        print("Iteration:", 0)
        for it in range(self.max_iter):
            sys.stdout.write("\033[F")
            print("Iteration:", it + 1)

            if rospy.is_shutdown():
                print("[INFO] ROS shutdown. Stopping RRT.")
                break

            # 1) Sample a configuration
            rand_config = self.config_space.sample_config(goal)
            if self.config_space.check_collision(rand_config):
                continue

            # 2) Find nearest node
            closest_config = self.config_space.nearest_config_to(self.graph.nodes, rand_config)

            # 3) Attempt a local plan
            path = self.config_space.local_plan(closest_config, rand_config, dt=dt)
            if path is None:
                continue

            # 4) Check collisions with sub-stepping
            if self.config_space.check_path_collision(path):
                # If we collide, let's try a smaller local_T adaptively
                smaller_path = self.config_space.adaptive_local_plan(closest_config, rand_config, dt=dt)
                if smaller_path is None or self.config_space.check_path_collision(smaller_path):
                    continue
                # Use smaller_path if valid
                path = smaller_path

            # 5) Only take a prefix
            delta_path = path.get_prefix(prefix_time_length)
            new_config = delta_path.end_position()

            # 6) Add node
            self.graph.add_node(new_config, closest_config, delta_path)

            # 7) Check distance to goal
            if self.config_space.distance(new_config, goal) <= self.expand_dist:
                # Try connecting to goal
                path_to_goal = self.config_space.local_plan(new_config, goal, dt=dt)
                if path_to_goal and not self.config_space.check_path_collision(path_to_goal):
                    self.graph.add_node(goal, new_config, path_to_goal)
                    self.plan = self.graph.construct_path_to(goal)
                    return self.plan

        print("Failed to find plan in allotted number of iterations.")
        return None

    def plot_execution(self):
        """
        Plots the RRT graph and final plan in the x-y plane.
        """
        ax = plt.subplot(1, 1, 1)
        ax.set_aspect(1)
        ax.set_xlim(self.config_space.low_lims[0], self.config_space.high_lims[0])
        ax.set_ylim(self.config_space.low_lims[1], self.config_space.high_lims[1])

        # Plot obstacles
        for obs in self.config_space.obstacles:
            xc, yc, r = obs
            circle = plt.Circle((xc, yc), r, color='black')
            ax.add_artist(circle)

        # Plot edges
        if self.graph:
            for path in self.graph.get_edge_paths():
                xs = path.positions[:, 0]
                ys = path.positions[:, 1]
                ax.plot(xs, ys, color='orange')

        # Plot final plan
        if self.plan:
            px = self.plan.positions[:, 0]
            py = self.plan.positions[:, 1]
            ax.plot(px, py, color='green')

        plt.show()



