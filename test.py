# -*- coding: utf-8 -*-
# @Time    : 2022/12/16 14:04
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation


class Field:
    num = 0

    def __init__(self, scope=(lambda x, y, z: True), b=None):
        if b is None:
            b = [0, 0, 0]
        self.scope = scope
        self.B = b
        self.num = Field.num
        Field.num += 1

    def add_scope(self, new_scope: iter, mode: bool):
        if mode:
            self.scope = lambda x, y, z: self.scope(x, y, z) and new_scope(x, y, z)
        else:
            self.scope = lambda x, y, z: self.scope(x, y, z) or new_scope(x, y, z)


class Particle:
    num = 0

    def __init__(self, m=0, q=0, v=None, f=None, position=None, color=None):
        if v is None:
            v = [0, 0, 0]
        if position is None:
            position = [0, 0, 0]
        if f is None:
            f = dict()
        self.position = np.array(position, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)
        self.m = m
        self.q = q
        self.f = f
        self.color = color
        for f in self.f:
            self.f[f] = np.array(self.f[f], dtype=np.float64)
        self.num = Particle.num
        Particle.num += 1


class Space:
    def __init__(self, interval):
        self.interval = interval
        self.fields = [
            Field(b=[0, 0, -1]),
        ]
        self.particles = [
            Particle(m=1, v=[0, 10, 0], f={'G': [0, 0, -10]}, q=10, position=[0, 0, 10], color='green'),
        ]

    def begin(self, t):
        path = [[[], [], []] for _ in self.particles]
        ax = plt.axes(projection='3d')
        particles = self.particles.copy()
        for i in range(t):
            for j in particles:
                for k in self.fields:
                    if k.scope(*j.position):
                        j.f[k.num] = np.cross(j.v, k.B) * j.q
                    else:
                        j.f[k.num] = 0
                if j.position[2] < 0:
                    particles.remove(j)
                    break
                dv = np.array([0, 0, 0], dtype=np.float64)
                for f in j.f:
                    dv += j.f[f] / j.m * self.interval
                j.v += dv
                j.position += j.v * self.interval
                path[j.num][0].append(j.position[0])
                path[j.num][1].append(j.position[1])
                path[j.num][2].append(j.position[2])
        for n, i in enumerate(path):
            ax.plot3D(*i, color=self.particles[n].color)
        return


def main():
    Space(0.01).begin(100)
    plt.show()


if __name__ == '__main__':
    main()
