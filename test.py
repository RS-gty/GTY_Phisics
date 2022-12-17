# -*- coding: utf-8 -*-
# @Time    : 2022/12/16 14:04
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt


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

    def __str__(self):
        return f'Field {self.num}'

    def __repr__(self):
        return f'Field {self.num}'


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
        self.track = [self.position.copy()]
        self.v = np.array(v, dtype=np.float64)
        self.m = m
        self.q = q
        self.f = f
        self.color = color
        for f in self.f:
            self.f[f] = np.array(self.f[f], dtype=np.float64)
        self.num = Particle.num
        Particle.num += 1

    def move(self, interval):
        for f in self.f:
            self.v += self.f[f] / self.m * interval
        self.position += self.v * interval
        self.track.append(self.position.copy())

    def __str__(self):
        return f'Particle {self.num}'

    def __repr__(self):
        return f'Particle {self.num}'


class Space:
    def __init__(self, interval, fields=None, particles=None):
        if particles is None:
            particles = []
        if fields is None:
            fields = []
        self.interval = interval
        self.fields = fields
        self.particles = particles

    def begin(self, t):
        ax = plt.axes(projection='3d')
        for _ in range(t):
            for p in self.particles:
                if p.position[2] < 0:
                    continue
                for k in self.fields:
                    if k.scope(*p.position):
                        p.f[k.num] = np.cross(p.v, k.B) * p.q
                    else:
                        p.f[k.num] = 0
                p.move(self.interval)
        for p in self.particles:
            ax.plot3D(*np.array(p.track).T, c=p.color)
        return

    def __str__(self):
        return 'Space'

    def __repr__(self):
        return 'Space'


def main():
    fields = [
        Field(b=[0, 0, -1]),
    ]
    particles = [
        Particle(m=1, v=[0, 2, 10], f={'G': [0, 0, -10]}, q=10, position=[0, 0, 0], color='red'),
    ]
    Space(0.001, fields, particles).begin(10000)
    plt.show()


if __name__ == '__main__':
    main()
