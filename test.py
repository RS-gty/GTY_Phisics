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


if __name__ == '__main__':
    print(orthogonal_decomposition(np.array([1, 2, 3]), np.array([4, 2, -1])))
