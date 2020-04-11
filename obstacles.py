import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import patches

from klampt import *
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.math import vectorops

import random
import numpy as np

class Circle:
    """A circle geometry.  Can collide with circles or rectangles."""
    def __init__(self,center,radius):
        self.center = center
        self.radius = radius
    def contains(self,p):
        return vectorops.distance(self.center,p) <= self.radius
    def collides(self,other):
        if isinstance(other,Circle):
            return vectorops.distance(self.center,other.center) <= self.radius+other.radius
        elif isinstance(other,Rectangle):
            return other.collides(self)
        else:
            raise ValueError("Circle can only collide with Circle or Rectangle")
    def draw_matplotlib(self,ax,**args):
        ax.add_patch(patches.Circle(self.center,self.radius,**args))

class Rectangle:
    def __init__(self,bmin,bmax):
        self.bmin = bmin
        self.bmax = bmax
    def contains(self,p):
        return (self.bmin[0] <= p[0] <= self.bmax[0]) and (self.bmin[1] <= p[1] <= self.bmax[1])
    def collides(self,other):
        if isinstance(other,Circle):
            closest = (max(self.bmin[0],min(other.center[0],self.bmax[0])),
                       max(self.bmin[1],min(other.center[1],self.bmax[1])))
            return other.contains(closest)
        elif isinstance(other,Rectangle):
            return (other.bmin[0] <= self.bmin[0] != other.bmax[0] <= self.bmax[0]) and (other.bmin[1] <= self.bmin[1] != other.bmax[1] <= self.bmax[1])
        else:
            raise ValueError("Rectangle can only collide with Circle or Rectangle")
    def draw_matplotlib(self,ax,**args):
        ax.add_patch(patches.Rectangle(self.bmin,self.bmax[0]-self.bmin[0],self.bmax[1]-self.bmin[1],**args))

def create_obstacles(num):
    obstacles_collection = []
    for i in range(num):
        obstacles = []
        num_obs = random.randint(1,10)
        for j in range(num_obs):
            if random.randint(0,1):
                obstacles.append(Rectangle([round(random.random(), 3),round(random.random(), 3)],[round(random.random(), 3),round(random.random(), 3)]))
            else:
                rand_center = (round(random.random(), 3), round(random.random(), 3))
                radius_limit = [rand_center[0], 1 - rand_center[0], rand_center[1], 1 - rand_center[1]]
                rand_radius = round(random.uniform(0, min(radius_limit)), 3)
                obstacles.append(Circle(rand_center,rand_radius))
        obstacles_collection.append(obstacles)
    return obstacles_collection

if __name__ == "__main__":
    obstacles_collection = create_obstacles(5)
    # narrow_passage_obstacles = [Rectangle([0.3,0],[0.7,0.4]),Rectangle([0.3,0.6],[0.7,1.0])]
    count = 0
    for obstacles in obstacles_collection:
        plt.figure(figsize=(12,12))
        plt.axis('equal')
        plt.xlim(0,1)
        plt.ylim(0,1)
        for o in obstacles:
            o.draw_matplotlib(plt.gca(),color='k')
        plt.savefig(str(count).zfill(3)+'.png')
        count += 1
