import numpy as np

# globals

g = {'x': 0, 'y': 0}


# Calculations


def orthogonal_decomposition(vector, b1):
    v_parallel = b1 * (np.linalg.norm(vector) * np.dot(vector, b1) / (
            np.linalg.norm(vector) * np.linalg.norm(b1))) / np.linalg.norm(b1)
    return np.subtract(vector, v_parallel)


class Field(object):
    def __init__(self):
        self.con = []
        self.con_type = 0  # 0->and;1->or
        self.facing = np.array([0, 0, 0])
        self.strength = 0
        self.type = 0  # 0->Magnetic;1->Electric;2->Gravity_like

    def initial(self, field_type, strength, facing: np.ndarray):
        self.type = field_type
        self.strength = strength
        self.facing = facing

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
        self.position = np.array([0, 0, 0])
        self.speed = np.array([0, 0, 0])
        self.speed_amount = 0
        self.speed_bias = 0
        self.charge = 1
        self.mass = 1
        self.force = np.array([0, 0, 0])
        self.force_amount = 0
        self.acceleration = np.array([0, 0, 0])


    def initial(self, speed):
        self.speed = speed
        self.speed_amount = np.linalg.norm(self.speed)
        self.speed_bias = np.linalg.norm(self.speed)

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
                pass
            else:
                vec = (f.facing * (1 / np.linalg.norm(f.facing)) * f.electric_intensity * self.charge)[0: 2]
            temp.append(vec)
        for ts in temp:
            s_force = np.add(s_force, ts)
        self.force = s_force
        self.force_amount = np.linalg.norm(self.force)

    def update(self, dt, *field: Field):
        n = 0
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

        self.speed = self.speed_bias * self.speed / self.speed_amount
        self.speed_amount = np.linalg.norm(self.speed)
        if len(self.speed) == 1:
            self.speed = self.speed[0]
        else:
            pass


if __name__ == '__main__':
    print(orthogonal_decomposition(np.array([1, 2, 3]), np.array([4, 2, -1])))
