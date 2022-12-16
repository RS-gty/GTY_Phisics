import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

g = {'x': 0, 'y': 0}
n = 0


class Field(object):
    def __init__(self):
        self.con = []
        self.con_type = 0  # 0->and;1->or
        self.facing = np.array([0, 0, 0])
        self.flux_density = 0
        self.electric_intensity = 0
        self.type = 0  # 0->Magnetic;1->Electric

    def initial(self, field_type, strength, facing: np.ndarray):
        self.facing = facing
        self.type = field_type
        if field_type == 1:
            self.electric_intensity = strength
        else:
            self.flux_density = strength

    def check(self, x, y):
        g['x'] = x
        g['y'] = y
        if self.con_type == 0:
            for p in self.con:
                if eval(p, g):
                    pass
                else:
                    return False
            return True
        else:
            for p in self.con:
                if eval(p, g):
                    return True
                else:
                    pass
            return False


class ChargedParticle(object):
    def __init__(self):
        self.position = np.array([0, 0])
        self.speed = np.array([0, 0])
        self.speed_amount = 0
        self.charge = 1
        self.mass = 1
        self.force = np.array([0, 0])
        self.force_amount = 0
        self.acceleration = np.array([0, 0])
        self.speed_constant = 0

    def initial(self, speed):
        self.speed = speed
        self.speed_constant = np.linalg.norm(self.speed)

    def get_force(self, *field):
        temp = []
        s_force = 0
        s_field = []
        vec = 0
        if type(field[0]) == list:
            s_field = field[0]
        else:
            s_field = field

        for f in s_field:
            if f.type == 0:
                try:
                    vec = np.dot(np.linalg.inv(np.array([f.facing.tolist(),
                                                         self.speed.tolist() + [0],
                                                         [1, 0, 0]])), np.array([[0], [0], [1]]))[0: 2].T
                except np.linalg.LinAlgError:
                    vec = np.dot(np.linalg.inv(np.array([f.facing.tolist(),
                                                         self.speed.tolist() + [0],
                                                         [0, 1, 0]])), np.array([[0], [0], [1]]))[0: 2].T
                vec = vec * ((self.charge * np.linalg.norm(self.speed) * f.flux_density) / np.linalg.norm(vec))
                v_z = np.cross(self.speed, vec)
                if v_z[0] * f.facing[2] > 0:
                    vec = -vec
                else:
                    pass
            else:
                vec = (f.facing * (1 / np.linalg.norm(f.facing)) * f.electric_intensity * self.charge)[0: 2]
            temp.append(vec)
        for ts in temp:
            s_force = np.add(s_force, ts)
        self.force = s_force
        self.force_amount = np.linalg.norm(self.force)

    def update(self, dt, *field: Field):
        global n
        n += 1
        s_pos = self.position
        activate_field = []
        s_field = []
        if type(field[0]) == list:
            s_field = field[0]
        else:
            s_field = field

        if len(self.position) == 1:
            s_pos = self.position[0]
        else:
            pass

        for f in s_field:
            if f.check(s_pos[0], s_pos[1]):
                activate_field.append(f)
            else:
                pass

        if len(activate_field) != 0:
            self.get_force(activate_field)
            self.acceleration = self.force / self.mass
            self.speed = np.add(self.speed, self.acceleration * dt)
            self.position = np.add(self.position, self.speed * dt)
            self.speed_amount = np.linalg.norm(self.speed)
        else:
            self.force = 0
            self.force_amount = 0
            self.position = np.add(self.position, self.speed * dt)
            self.speed_amount = np.linalg.norm(self.speed)

        self.speed = self.speed_constant * self.speed / self.speed_amount
        self.speed_amount = np.linalg.norm(self.speed)
        if len(self.speed) == 1:
            self.speed = self.speed[0]
        else:
            pass


class Situation(object):
    def __init__(self):
        self.particles = []
        self.fields = []

    def initial(self, particles: list, fields: list):
        self.particles = particles
        self.fields = fields

    def simulate(self, total_time, dt=0.001):
        p_num = len(self.particles)
        f_num = len(self.fields)

        for p in self.particles:
            px = []
            py = []
            for i in tqdm(range(int(total_time / dt))):
                p.update(dt, self.fields)
                try:
                    px.append(p.position[0][0])
                    py.append(p.position[0][1])
                except IndexError:
                    px.append(p.position[0])
                    py.append(p.position[1])
            plt.plot(px, py)
        plt.gca().set_aspect('equal', 'box')
        plt.show()


if __name__ == '__main__':

    F1 = Field()
    F1.initial(0, 1, np.array([0, 0, 1]))
    F1.con = ['y>x*1.7320508', 'y>-x*1.7320508', 'y<-1.7320508/2']
    F1.con_type = 1

    F2 = Field()
    F2.initial(0, 1, np.array([0, 0, -1]))
    F2.con = ['y<x*1.7320508', 'y<-x*1.7320508', 'y>-1.7320508/2']
    F2.con_type = 0

    G = Field()
    G.initial(0, 1, np.array([0, 0, 1]))

    P1 = ChargedParticle()
    P1.initial(np.array([0, -1]))
    P2 = ChargedParticle()
    P2.initial(np.array([1, 0]))
    P3 = ChargedParticle()
    P3.initial(np.array([2, -0.8]))
    P4 = ChargedParticle()
    P4.initial(np.array([-1, -3]))

    S1 = Situation()
    S1.initial([P1], [F1])
    S1.simulate(10, 0.001)
