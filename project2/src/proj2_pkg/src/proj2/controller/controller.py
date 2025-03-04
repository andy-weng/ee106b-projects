#!/usr/bin/env python

import numpy as np
import rospy
from std_srvs.srv import Empty as EmptySrv
from proj2_pkg.msg import BicycleCommandMsg, BicycleStateMsg
from proj2.planners import SinusoidPlanner, RRTPlanner, BicycleConfigurationSpace

class BicycleModelController(object):
    def __init__(self):
        self.pub = rospy.Publisher('/bicycle/cmd_vel', BicycleCommandMsg, queue_size=10)
        self.sub = rospy.Subscriber('/bicycle/state', BicycleStateMsg, self.subscribe)
        self.state = np.array([0, 0, 0, 0])
        self.last_time = -1
        rospy.on_shutdown(self.shutdown)
        self.kp_x = 1.2 
        self.kd_x = 0.15 
        self.kp_phi = 1.5
        self.kd_phi = 0.2
        self.max_u1 = 1.5  
        self.max_u2 = 0.8 
        self.goal_tolerance_xy = 0.05 
        self.goal_tolerance_theta = 0.02

    def execute_plan(self, plan):
        if plan is None or len(plan) == 0:
            print("[ERROR] Plan is empty. Cannot execute.")
            return
        print("[INFO] Executing Plan...")
        rate = rospy.Rate(int(1 / plan.dt))
        start_t = rospy.Time.now()
        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_t).to_sec()
            if t > plan.times[-1]:
                print("[INFO] Reached end of plan. Stopping.")
                break
            target_state, open_loop_input = plan.get(t)
            self.step_control(target_state, open_loop_input)
            if self.is_goal_reached(target_state):
                print("[INFO] Goal reached. Stopping.")
                break
            rate.sleep()
        self.cmd([0, 0]) 

    def step_control(self, target_position, open_loop_input):
        error_x = target_position[0] - self.state[0]
        error_y = target_position[1] - self.state[1]
        error_phi = target_position[3] - self.state[3]
        error_xy = np.sqrt(error_x**2 + error_y**2)
        error_theta = np.arctan2(np.sin(target_position[2] - self.state[2]),
                                 np.cos(target_position[2] - self.state[2]))
        curr_time = rospy.Time.now()
        if self.last_time == -1:
            self.last_time = curr_time
            self.last_xy_error = error_xy
            self.last_phi_error = error_phi
            d_error_xy = 0.0
            d_error_phi = 0.0
        else:
            dt = (curr_time - self.last_time).to_sec()
            d_error_xy = (error_xy - self.last_xy_error) / dt
            d_error_phi = (error_phi - self.last_phi_error) / dt
            self.last_time = curr_time
            self.last_xy_error = error_xy
            self.last_phi_error = error_phi
        u1 = open_loop_input[0] + self.kp_x * error_xy + self.kd_x * d_error_xy
        u2 = open_loop_input[1] + self.kp_phi * error_theta + self.kd_phi * d_error_phi
        u1 = np.clip(u1, -self.max_u1, self.max_u1)
        u2 = np.clip(u2, -self.max_u2, self.max_u2)
        if self.is_goal_near(target_position):
            u1 *= 0.5
            u2 *= 0.5
        self.cmd([u1, u2])

    def is_goal_reached(self, target_position):
        error_x = abs(target_position[0] - self.state[0])
        error_y = abs(target_position[1] - self.state[1])
        error_theta = abs(np.arctan2(np.sin(target_position[2] - self.state[2]),
                                     np.cos(target_position[2] - self.state[2])))
        if error_x < self.goal_tolerance_xy and error_y < self.goal_tolerance_xy and error_theta < self.goal_tolerance_theta:
            return True
        return False

    def is_goal_near(self, target_position):
        return np.linalg.norm(target_position[:2] - self.state[:2]) < 0.3 

    def cmd(self, msg):
        """Publishes velocity commands to the robot."""
        self.pub.publish(BicycleCommandMsg(*msg))

    def subscribe(self, msg):
        """Updates the robot's current state from sensor data."""
        self.state = np.array([msg.x, msg.y, msg.theta, msg.phi])

    def shutdown(self):
        """Stops the robot when shutting down."""
        rospy.loginfo("Shutting Down")
        self.cmd([0, 0])

