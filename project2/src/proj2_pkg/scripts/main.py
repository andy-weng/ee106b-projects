#!/usr/bin/env python
"""
Starter code for EE106B Turtlebot Lab
Author: Valmik Prabhu, Chris Correa
Adapted for Spring 2020 by Amay Saxena

Re-implemented to tie everything together.
"""

import numpy as np
import sys
import argparse
from std_srvs.srv import Empty as EmptySrv
import rospy

from proj2_pkg.msg import BicycleCommandMsg, BicycleStateMsg
from proj2.planners import SinusoidPlanner, RRTPlanner, OptimizationPlanner, BicycleConfigurationSpace
from proj2.controller import BicycleModelController

def parse_args():
    """
    Parse command-line arguments for planner type and goal.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-planner', '-p', type=str, default='sin',
                        help='Options: sin, rrt, opt.  Default: sin')
    parser.add_argument('-x', type=float, default=1.0, help='Desired position in x')
    parser.add_argument('-y', type=float, default=1.0, help='Desired position in y')
    parser.add_argument('-theta', type=float, default=0.0, help='Desired turtlebot angle')
    parser.add_argument('-phi', type=float, default=0.0, help='Desired steering angle')
    return parser.parse_args()

if __name__ == '__main__':
    rospy.init_node('planning', anonymous=False)
    args = parse_args()

    # Reset robot state
    print('Waiting for converter/reset service ...')
    rospy.wait_for_service('/converter/reset')
    print('found!')
    reset = rospy.ServiceProxy('/converter/reset', EmptySrv)
    reset()

    # Load environment parameters
    if not rospy.has_param("/environment/obstacles"):
        raise ValueError("No environment info. Did you run init_env.launch?")
    obstacles = rospy.get_param("/environment/obstacles")

    if not rospy.has_param("/environment/low_lims"):
        raise ValueError("No environment info. Did you run init_env.launch?")
    xy_low = rospy.get_param("/environment/low_lims")

    if not rospy.has_param("/environment/high_lims"):
        raise ValueError("No environment info. Did you run init_env.launch?")
    xy_high = rospy.get_param("/environment/high_lims")

    if not rospy.has_param("/bicycle_converter/converter/max_steering_angle"):
        raise ValueError("No robot info. Did you run init_env.launch?")
    phi_max = rospy.get_param("/bicycle_converter/converter/max_steering_angle")

    if not rospy.has_param("/bicycle_converter/converter/max_steering_rate"):
        raise ValueError("No robot info. Did you run init_env.launch?")
    u2_max = rospy.get_param("/bicycle_converter/converter/max_steering_rate")

    if not rospy.has_param("/bicycle_converter/converter/max_linear_velocity"):
        raise ValueError("No robot info. Did you run init_env.launch?")
    u1_max = rospy.get_param("/bicycle_converter/converter/max_linear_velocity")

    print("Obstacles:", obstacles)
    
    controller = BicycleModelController()
    rospy.sleep(1)

    print("Initial State")
    print(controller.state)

    goal = np.array([args.x, args.y, args.theta, args.phi])

    # Build the bicycle configuration space
    config = BicycleConfigurationSpace(
        xy_low + [-1000, -phi_max],
        xy_high + [1000, phi_max],
        [-u1_max, -u2_max],
        [u1_max, u2_max],
        obstacles,
        0.15
    )

    # Choose planner
    if args.planner == 'sin':
        planner = SinusoidPlanner(config)
        plan = planner.plan_to_pose(controller.state, goal, dt=0.01, delta_t=2.0)
    elif args.planner == 'rrt':
        planner = RRTPlanner(config, max_iter=15000, expand_dist=0.3)
        plan = planner.plan_to_pose(controller.state, goal, dt=0.01, prefix_time_length=0.4)
    elif args.planner == 'opt':
        planner = OptimizationPlanner(config)
        plan = planner.plan_to_pose(controller.state, goal, dt=0.01, N=500)
    else:
        raise ValueError("Unknown planner type. Use sin, rrt, or opt.")

    if plan is None:
        print("[ERROR] Planner failed to find a plan.")
        sys.exit(1)

    print("Predicted Initial State:", plan.start_position())
    print("Predicted Final State:", plan.end_position())

    planner.plot_execution()

    # Execute the plan
    controller.execute_plan(plan)
    print("Final State:", controller.state)

