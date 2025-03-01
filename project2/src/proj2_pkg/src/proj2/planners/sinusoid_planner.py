#!/usr/bin/env python
"""
Starter code for EE106B Turtlebot Lab
Author: Valmik Prabhu, Chris Correa
Adapted for Spring 2020 by Amay Saxena
"""
import numpy as np
from scipy.integrate import quad
import sys
from copy import copy
import matplotlib.pyplot as plt
from .configuration_space import Plan, BicycleConfigurationSpace

class SinusoidPlanner():
    def __init__(self, config_space):
        """
        Turtlebot planner that uses sequential sinusoids to steer to a goal pose.

        config_space should be a BicycleConfigurationSpace object.
        Parameters
        ----------
        l : float
            length of car
        """
        self.config_space = config_space
        self.l = config_space.robot_length
        self.max_phi = config_space.high_lims[3]
        self.max_u1 = config_space.input_high_lims[0]
        self.max_u2 = config_space.input_high_lims[1]

    def plan_to_pose(self, start_state, goal_state, t0=0.0, dt=0.01, delta_t=2.0, alpha_segments=2, y_segments=2):
        """
        Plans to a specific pose in (x,y,theta,phi) coordinates.  You 
        may or may not have to convert the state to a v state with state2v()
        You may want to plan each component separately
        so that you can reset phi in case there's drift in phi.

        You will need to edit some or all of this function to take care of
        configuration

        Parameters
        ----------
        start_state: numpy.ndarray of shape (4,) [x, y, theta, phi]
        goal_state: numpy.ndarray of shape (4,) [x, y, theta, phi]
        dt : float
            how many seconds between trajectory timesteps
        delta_t : float
            how many seconds each trajectory segment should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """

        print("======= Planning with SinusoidPlanner =======")

        self.plan = None
        # This bit hasn't been exhaustively tested, so you might hit a singularity anyways
        x_s, y_s, theta_s, phi_s = start_state
        x_g, y_g, theta_g, phi_g = goal_state
        max_abs_angle = max(abs(theta_g), abs(theta_s))
        min_abs_angle = min(abs(theta_g), abs(theta_s))
        if (max_abs_angle > np.pi/2) and (min_abs_angle < np.pi/2):
            raise ValueError("You'll cause a singularity here. You should add something to this function to fix it")

        if abs(phi_s) > self.max_phi or abs(phi_g) > self.max_phi:
            raise ValueError("Either your start state or goal state exceeds steering angle bounds")

        # We can only change phi up to some threshold
        self.phi_dist = min(
            abs(phi_g - self.max_phi),
            abs(phi_g + self.max_phi)
        )

        x_path =        self.steer_x(
                            start_state, 
                            goal_state, 
                            t0,
                            dt=dt, 
                            delta_t=delta_t
                        )
        phi_start_time = x_path.times[-1]
        phi_path =      self.steer_phi(
                            x_path.end_position(), 
                            goal_state,  
                            t0=phi_start_time,
                            dt=dt, 
                            delta_t=delta_t
                        )
        alpha_start_time = phi_path.times[-1]
        alpha_path = self.plan_alpha_in_segments(
                            phi_path.end_position(),
                            goal_state,
                            t0=alpha_start_time,
                            dt=dt,
                            delta_t=delta_t,
                            n=alpha_segments
                        )
        
        y_start_time = alpha_path.times[-1]
        # y_path =        self.plan_y_in_segments(
        #                     alpha_path.end_position(), 
        #                     goal_state,
        #                     t0=y_start_time,
        #                     dt=dt,
        #                     delta_t=delta_t,
        #                     n = y_segments
        #                 )     
        y_path =        self.steer_y(
                            alpha_path.end_position(), 
                            goal_state,
                            t0=y_start_time,
                            dt=dt,
                            delta_t=delta_t,
                        )  
        self.plan = Plan.chain_paths(x_path, phi_path, alpha_path, y_path)
        return self.plan
    
    def plan_alpha_in_segments(self, start_state, goal_state,
                               t0=0.0, dt=0.01, delta_t=2.0, n=2):
       
        partial_plans = []
        current_state = start_state
        current_time  = t0

        start_v = self.state2v(start_state)  
        goal_v  = self.state2v(goal_state)
        alpha_s = start_v[2]
        alpha_g = goal_v[2]

        for i in range(1, n+1):
            frac      = i / float(n)
            alpha_sub = alpha_s + frac*(alpha_g - alpha_s)

            sub_goal_v = copy(goal_v)
            sub_goal_v[2] = alpha_sub 
            sub_goal_state = self.v2state(sub_goal_v)

            sub_path = self.steer_alpha(current_state,
                                        sub_goal_state,
                                        t0=current_time,
                                        dt=dt, delta_t=delta_t)

            partial_plans.append(sub_path)
            current_state = sub_path.end_position()
            current_time  = sub_path.times[-1]
        print(partial_plans)
        return Plan.chain_paths(*partial_plans)
    
    def plan_y_in_segments(self, start_state, goal_state,
                           t0=0.0, dt=0.01, delta_t=2.0, n=2):
        
        partial_plans = []
        current_state = start_state
        current_time  = t0

        start_v = self.state2v(start_state)  
        goal_v  = self.state2v(goal_state)
        y_s = start_v[3]
        y_g = goal_v[3]

        for i in range(1, n+1):
            frac = i / float(n)
            y_sub = y_s + frac*(y_g - y_s)

            sub_goal_v = copy(goal_v)
            sub_goal_v[3] = y_sub
            sub_goal_state = self.v2state(sub_goal_v)

            sub_plan = self.steer_y(current_state,
                                    sub_goal_state,
                                    t0=current_time,
                                    dt=dt, delta_t=delta_t)

            partial_plans.append(sub_plan)
            current_state = sub_plan.end_position()
            current_time  = sub_plan.times[-1]

        return Plan.chain_paths(*partial_plans)

    def plot_execution(self):
        """
        Creates a plot of the planned path in the environment. Assumes that the 
        environment of the robot is in the x-y plane, and that the first two
        components in the state space are x and y position. Also assumes 
        plan_to_pose has been called on this instance already, so that self.graph
        is populated. If planning was successful, then self.plan will be populated 
        and it will be plotted as well.
        """
        ax = plt.subplot(1, 1, 1)

        if self.plan:
            plan_x = self.plan.positions[:, 0]
            plan_y = self.plan.positions[:, 1]
            ax.plot(plan_x, plan_y, color='green')

        plt.show()

    def steer_x(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a Plan to move the turtlebot in the x direction

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state)
        goal_state_v = self.state2v(goal_state)
        delta_x = goal_state_v[0] - start_state_v[0]

        v1 = delta_x/delta_t
        v2 = 0

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1, v2])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt)

    def steer_phi(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a trajectory to move the turtlebot in the phi direction

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state)
        goal_state_v = self.state2v(goal_state)
        delta_phi = goal_state_v[1] - start_state_v[1]

        v1 = 0
        v2 = delta_phi/delta_t

        path,t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1, v2])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt)


    def steer_alpha(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a trajectory to move the turtlebot in the alpha direction.  
        Remember dot{alpha} = f(phi(t))*u_1(t) = f(frac{a_2}{omega}*sin(omega*t))*a_1*sin(omega*t)
        also, f(phi) = frac{1}{l}tan(phi)
        See the doc for more math details

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """

        start_state_v = self.state2v(start_state)
        goal_state_v = self.state2v(goal_state)
        delta_alpha = goal_state_v[2] - start_state_v[2]

        omega = 2*np.pi / delta_t

        a2 = min(1, self.phi_dist*omega)
        f = lambda phi: (1/self.l)*np.tan(phi) # This is from the car model
        phi_fn = lambda t: (a2/omega)*np.sin(omega*t) + start_state_v[1]
        integrand = lambda t: f(phi_fn(t))*np.sin(omega*t) # The integrand to find beta
        beta1 = (omega/np.pi) * quad(integrand, 0, delta_t)[0]

        a1 = (delta_alpha*omega)/(np.pi*beta1)
              
        v1 = lambda t: a1*np.sin(omega*(t))
        v2 = lambda t: a2*np.cos(omega*(t))

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1(t-t0), v2(t-t0)])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt)


    def steer_y(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a trajectory to move the turtlebot in the y direction. 
        Remember, dot{y} = g(alpha(t))*v1 = frac{alpha(t)}{sqrt{1-alpha(t)^2}}*a_1*sin(omega*t)
        See the doc for more math details

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state)
        goal_state_v = self.state2v(goal_state)
        delta_y = goal_state_v[3] - start_state_v[3]

        omega = 2*np.pi / delta_t

        a2 = min(1, self.phi_dist*omega)

        f = lambda phi: (1/self.l)*np.tan(phi) # This is from the car model
        phi_fn = lambda t: (a2/omega)*np.sin(omega*t) + start_state_v[1]
        integrand_phi = lambda t: f(phi_fn(t))*np.sin(omega*t) # The integrand to find beta
        
        g = lambda alpha: alpha/np.sqrt(1 - alpha**2)
        alpha_fn = lambda t,a1: quad(integrand_phi,0,t,(a1)) + start_state_v[2]
        integrand = lambda t, a1: a1*g(alpha_fn(t,a1)) * np.sin(omega * t)

        a1_low = 0
        a1_high = self.max_u1
        tol = 0.05

        for i in range(100):  
            a1 = (a1_low + a1_high) / 2  
            beta1 = (omega / np.pi) * quad(integrand, 0, delta_t, (a1))[0]  

            guess_y = a1 * (np.pi / omega) * beta1 
            print (beta1)

            if np.abs(guess_y - delta_y) < tol:
                print("done")
                break  
            elif guess_y < delta_y:               
                a1_low = a1  
            else:
                a1_high = a1  
            
        
        v1 = lambda t: a1*np.sin(omega*(t))
        v2 = lambda t: a2*np.cos(2*omega*(t))

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1(t-t0), v2(t-t0)])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt)

    def state2v(self, state):
        """
        Takes a state in (x,y,theta,phi) coordinates and returns a state of (x,phi,alpha,y)

        Parameters
        ----------
        state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            some state

        Returns
        -------
        4x1 :obj:`numpy.ndarray` 
            x, phi, alpha, y
        """
        x, y, theta, phi = state
        return np.array([x, phi, np.sin(theta), y])
    
    def v2state(self, v):
        """
        Convert [x, phi, alpha, y] -> (x,y,theta,phi).
        Inverse of state2v assuming alpha = sin(theta).
        """
        x, phi, alpha, y = v
        # Handle alpha -> theta
        # If alpha = sin(theta), pick a principal solution. 
        # (This can be tricky if alpha in [-1,1], but let's do a simple arcsin.)
        theta = np.arcsin(alpha)
        return np.array([x, y, theta, phi])

    def state2v1(self,state):
        x,y,theta,phi = state
        return np.array([y,phi,np.sin(theta),x])

    def v_path_to_u_path(self, path, start_state, dt):
        """
        convert a trajectory in v commands to u commands

        Parameters
        ----------
        path : :obj:`list` of (float, float, float)
            list of (time, v1, v2) commands
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            starting state of this trajectory
        dt : float
            how many seconds between timesteps in the trajectory

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        def v2cmd(v1, v2, state):
            u1 = v1/np.cos(state[2])
            u2 = v2
            return [u1, u2]

        curr_state = start_state
        positions = []
        times = []
        open_loop_inputs = []
        for i, (t, v1, v2) in enumerate(path):
            cmd_u = v2cmd(v1, v2, curr_state)
            positions.append(curr_state)
            open_loop_inputs.append(cmd_u)
            times.append(t)

            x, y, theta, phi = curr_state
            linear_velocity, steering_rate = cmd_u
            curr_state = [
                x     + np.cos(theta)               * linear_velocity*dt,
                y     + np.sin(theta)               * linear_velocity*dt,
                theta + np.tan(phi) / float(self.l) * linear_velocity*dt,
                phi   + steering_rate*dt
            ]

        return Plan(np.array(times), np.array(positions), np.array(open_loop_inputs), dt=dt)

def main():
    """Use this function if you'd like to test without ROS.
    """
    start = np.array([1, 1, 0, 0]) 
    goal = np.array([1, 1, 0.3, 0])
    xy_low = [0, 0]
    xy_high = [5, 5]
    phi_max = 0.6
    u1_max = 2
    u2_max = 3
    obstacles = []

    config = BicycleConfigurationSpace( xy_low + [-1000, -phi_max],
                                        xy_high + [1000, phi_max],
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obstacles,
                                        0.15)

    planner = SinusoidPlanner(config)

    plan = planner.plan_to_pose(start, goal, 0.0, 0.1, 2.0, alpha_segments=1,y_segments=4)
    planner.plot_execution()

if __name__ == '__main__':
    main()
