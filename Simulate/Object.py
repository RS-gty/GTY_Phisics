from tqdm import tqdm
from Animate import *

# globals
g = {'x': 0, 'y': 0, 'z': 0}


# Force
class Force(object):
    """

    :param a: Strength of the force
    :param dr: Direction of the force
    """

    def __init__(self, a, dr: list):
        # Inner Attributes
        self.amount = np.float64(a)  # example:g ; B ; G*M2...
        self.direction = np.array(dr) / np.linalg.norm(np.array(dr))
        self.range = []
        self.range_type = 0  # 0 -> and; 1 -> or
        self.type = None

    def check(self, position: np.ndarray):
        g['x'] = position[0]
        g['y'] = position[1]
        g['z'] = position[2]

        if self.range_type == 0:
            for c in self.range:
                if eval(c, g):
                    pass
                else:
                    return False
            return True
        else:
            for c in self.range:
                if eval(c, g):
                    return True
                else:
                    pass
            return False

    def get_force(self, *args):
        pass


class ConstantForce(Force):
    """

    :param a: Strength of the force
    :param dr: Direction of the force
    """

    def __init__(self, a, dr):
        super().__init__(a, dr)
        self.type = 0

    def get_force(self, k=1):
        return self.amount * k * self.direction


class InverseSquareForce(Force):
    """

    :param a: Constant times property of the distance source
           (for gravity, G * m2)
    :param s: The position of the distance source
    """

    def __init__(self, a, s: list):
        super().__init__(a, [0, 0, 0])
        self.distance_source = np.array(s)
        self.type = 1

    def get_force(self, p: np.ndarray, k):
        r = np.subtract(self.distance_source, p)
        return (self.amount * k / np.linalg.norm(r) ** 3) * r

    def change_source(self, d: np.ndarray):
        self.distance_source = d


class LorentzForce(Force):
    """

    :param a: Strength of the force
    :param dr: Direction of the force
    """

    def __init__(self, a, dr):
        super().__init__(a, dr)
        self.type = 2

    def get_force(self, s: np.ndarray, k=1):
        return k * self.amount * np.cross(s, self.direction)


# Particle
class Particle(object):
    def __init__(self):
        # Inner Attributes
        self.charge = np.float64(0)
        self.mass = np.float64(1)
        # Dynamic Properties
        self.force = np.array([np.float64(0), np.float64(0), np.float64(0)])

        self.V_speed = np.array([np.float64(0), np.float64(0), np.float64(0)])
        self.speed = np.float64(0)

        self.position = np.array([np.float64(0), np.float64(0), np.float64(0)])
        #  Additions
        self.related_force = []
        self.collapse = 0

    def update(self, dt, force: list[Force], k=1):
        # register forces
        self.force = np.array([np.float64(0), np.float64(0), np.float64(0)])
        activate_force = []
        for f in force:
            if f.check(self.position):
                activate_force.append(f)

        # combine force
        for f in activate_force:
            if f in self.related_force:
                f.change_source(self.position)
            else:
                if f.type == 0:
                    self.force += f.get_force(k)
                elif f.type == 1:
                    if np.linalg.norm(np.subtract(self.position, f.distance_source)) >= 0.02:
                        self.force += f.get_force(self.position, k)
                    else:
                        self.collapse = 1
                elif f.type == 2:
                    self.force += f.get_force(self.V_speed, self.charge)

        # get acceleration & speed change and correction
        self.V_speed += (self.force / self.mass) * dt
        if self.speed == 0:
            pass
        self.speed = np.linalg.norm(self.V_speed)

        # position change
        self.position += self.V_speed * dt

        # related force update(ONLY FOR INVERSE_SQUARE_FORCE)


class Simulation(object):
    def __init__(self, particles: list, forces: list):
        self.particles = particles
        self.forces = forces

    def simulate(self, total_time, dt=0.001, show_force_source=False, show=0):
        for p in range(len(self.particles)):
            lc['Px' + str(p)] = []
            lc['Py' + str(p)] = []
            lc['Pz' + str(p)] = []

        for pa in self.particles:
            pa.collapse = 0

        for i in tqdm(range(int(total_time / dt))):
            index = 0
            for p in self.particles:
                if p.collapse == 0:
                    p.update(dt, self.forces)
                    lc['Px' + str(index)].append(p.position[0])
                    lc['Py' + str(index)].append(p.position[1])
                    lc['Pz' + str(index)].append(p.position[2])
                else:
                    lc['Px' + str(index)].append(lc['Px' + str(index)][-1])
                    lc['Py' + str(index)].append(lc['Py' + str(index)][-1])
                    lc['Pz' + str(index)].append(lc['Pz' + str(index)][-1])


                index += 1
        for index in range(len(self.particles)):
            lc['Path' + str(index)] = Path(lc['Px' + str(index)], lc['Py' + str(index)], lc['Pz' + str(index)], index)
        P_l = list(lc['Path'+str(j)] for j in range(len(self.particles)))
        Anim = Animation(P_l, show)
        if show_force_source:
            for f in self.forces:
                if type(f) == InverseSquareForce:
                    ax.scatter3D(f.distance_source[0], f.distance_source[1], f.distance_source[2])
        ax.view_init(elev=1, azim=33)
        Anim.animate()
