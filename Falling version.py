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

    def exert(self, particle):
        particle.f[f'Lorentz force from field@{self.num}'] = np.cross(particle.v, self.B) * particle.q \
            if self.scope(*particle.position) else 0

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
        self.activation = True
        for f in self.f:
            self.f[f] = np.array(self.f[f], dtype=np.float64)
        self.num = Particle.num
        Particle.num += 1

    def move(self, interval):
        for f in self.f:
            self.v += self.f[f] / self.m * interval
        self.position += self.v * interval
        self.track.append(self.position.copy())

    def g_exert(self, particle, big_g):
        v = self.position - particle.position
        r = np.linalg.norm(v)
        particle.f[f'G from particle:{self.num}'] = v * big_g * self.m * particle.m / r ** 2 * np.linalg.norm(v) \
            if r else 0

    def e_exert(self, particle, k):
        v = particle.position - self.position
        r = np.linalg.norm(v)
        particle.f[f'Coulomb force from particle:{self.num}'] =\
            v * k * self.q * particle.q / r ** 2 * np.linalg.norm(v) if r else 0

    def __repr__(self):
        return f'Particle {self.num}'


class Space:
    def __init__(self, interval, fields=None, particles=None, big_g=1.0, k=1.0):
        if particles is None:
            particles = []
        if fields is None:
            fields = []
        self.interval = interval
        self.fields = fields
        self.particles = particles
        self.G = big_g
        self.k = k

    def start(self, t):
        ax = plt.axes(projection='3d')
        for _ in range(t):
            for p in self.particles:
                if not p.activation:
                    continue
                # elif p.position[2] < 0:
                #     p.activation = False
                #     continue
                for f in self.fields:
                    f.exert(p)
                for i in self.particles:
                    if i.activation:
                        i.g_exert(p, self.G)
                        i.e_exert(p, self.k)
            for p in self.particles:
                if p.activation:
                    p.move(self.interval)
        for p in self.particles:
            ax.plot3D(*np.array(p.track).T, c=p.color)

    def __repr__(self):
        return 'Space'


def main():
    fields = [
        # Field(b=[0, 0, 0]),
    ]
    particles = [
        Particle(m=1, v=[0, 0, 0], f={'g': [0, 0, 0]}, q=1, position=[0, 0, -1], color='red'),
        Particle(m=1, v=[0, 0, 0], f={'g': [0, 0, 0]}, q=-2, position=[1, 0, 0], color='green'),
        Particle(m=1, v=[0, 0, 0], f={'g': [0, 0, 0]}, q=-1, position=[0, -1, 0], color='blue'),
        Particle(m=1, v=[0, 0, 0], f={'g': [0, 0, 0]}, q=3, position=[1, -1, 0], color='yellow'),
    ]
    Space(0.0001, fields, particles, 0, 1).start(30000)
    plt.show()


if __name__ == '__main__':
    main()
