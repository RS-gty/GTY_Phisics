import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from mpl_toolkits import mplot3d

# globals

g = {'x': 0, 'y': 0, 'z': 0}


# Calculations


def orthogonal_decomposition(vector, b1):
    v_parallel = b1 * (np.linalg.norm(vector) * np.dot(vector, b1) / (
            np.linalg.norm(vector) * np.linalg.norm(b1))) / np.linalg.norm(b1)
    return np.subtract(vector, v_parallel)


class Field(object):
    def __init__(self, field_type, strength, facing: np.ndarray):
        self.con = []
        self.con_type = 0  # 0->and;1->or
        self.facing = facing
        self.strength = strength
        self.type = field_type  # 0->Magnetic;1->Electric;2->Gravity_like

    def check(self, position: np.ndarray):
        g['x'] = position[0]
        g['y'] = position[1]
        g['z'] = position[2]
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
    def __init__(self, speed, charge, mass):
        self.position = np.array([0, 0, 0])
        self.speed = speed
        self.speed_amount = np.linalg.norm(self.speed)
        self.speed_bias = np.linalg.norm(self.speed)
        self.charge = charge
        self.mass = mass
        self.force = np.array([0, 0, 0])
        self.force_amount = 0
        self.acceleration = np.array([0, 0, 0])
        self.collapse = 0

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
                vec = (self.speed_amount * f.strength * self.charge) * np.cross(self.speed, f.facing) / (
                        self.speed_amount * np.linalg.norm(f.facing))
            elif f.type == 1:
                vec = (f.facing * (1 / np.linalg.norm(f.facing)) * f.strength * self.charge)
            elif f.type == 2:
                f_facing = np.subtract(f.facing, self.position)
                if np.linalg.norm(f_facing) >= 0.01:
                    vec = self.mass * f.strength * f_facing / np.linalg.norm(f_facing) ** 3
                else:
                    self.collapse = 1

            temp.append(vec)
        for ts in temp:
            s_force = np.add(s_force, ts)
        self.force = s_force
        self.force_amount = np.linalg.norm(self.force)

    def update(self, dt, *field: Field):
        n = 0
        n += 1
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
            if f.check(self.position):
                activate_field.append(f)
            else:
                pass
        if self.collapse == 0:
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
        else:
            pass

        self.speed = self.speed_bias * self.speed / self.speed_amount
        self.speed_amount = np.linalg.norm(self.speed)
        if len(self.speed) == 1:
            self.speed = self.speed[0]
        else:
            pass


class Situation(object):
    def __init__(self, particles: list, fields: list):
        self.particles = particles
        self.fields = fields

    def simulate(self, total_time, dt=0.001):
        ax = plt.axes(projection='3d')
        for p in self.particles:
            px = []
            py = []
            pz = []
            for i in tqdm(range(int(total_time / dt))):
                p.update(dt, self.fields)
                try:
                    px.append(p.position[0][0])
                    py.append(p.position[0][1])
                    pz.append(p.position[0][2])
                except IndexError:
                    px.append(p.position[0])
                    py.append(p.position[1])
                    pz.append(p.position[2])
            ax.plot3D(px, py, pz)
        plt.gca().set_aspect('equal', 'box')
        plt.show()


if __name__ == '__main__':

    F1 = Field(0, 1, np.array([1, 0, 0]))
    F2 = Field(2, 1, np.array([0, 0, 1]))


    P1 = ChargedParticle(np.array([1, 0, 0]), 1, 1)
    P2 = ChargedParticle(np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]), 1, 1)
    P3 = ChargedParticle(np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]), 1, 1)
    P4 = ChargedParticle(np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]), 1, 1)
    P5 = ChargedParticle(np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]), 1, 1)
    P6 = ChargedParticle(np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]), 1, 1)
    P7 = ChargedParticle(np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]), 1, 1)
    S1 = Situation([P1], [F2])
    S1.simulate(30, 0.001)
    print('break')
