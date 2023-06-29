import matplotlib.lines
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# locals
lc = locals()

# globals

index_l = []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# functions

def convert1(m: np.ndarray):
    t = []
    for i in m:
        t.append(int(i))
    return np.array(t)


def limit(l: list[np.ndarray], n: int):
    """
    :param l: list
    :param n: 0 --- maximum ; 1 --- minimum
    """
    t = []
    for i in l:
        if n == 0:
            t.append(max(i))
        elif n == 1:
            t.append(min(i))
    if n == 0:
        return max(t)
    elif n == 1:
        return min(t)


# classes

class Path(object):
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, index: int):
        self.x = x
        self.y = y
        self.z = z
        self.px = []
        self.py = []
        self.pz = []
        lc['line' + str(index)], = plt.plot([], [], [], "r-", animated=True)
        self.line = [lc['line' + str(index)]]
        index_l.append(index)


class Animation(object):
    def __init__(self, paths: list[Path], s=0):
        """
        :param s: 0 --- track ; 1 --- real
        """
        self.particles = paths
        self.show = s

    def animate(self):

        # particles initialization

        P_list = self.particles

        # adjust limitations

        xm, ym, zm = [], [], []

        for p in P_list:
            xm.append(p.x)
            ym.append(p.y)
            zm.append(p.z)

        ax.set_xlim(limit(xm, 1), limit(xm, 0))
        ax.set_ylim(limit(ym, 1), limit(ym, 0))
        ax.set_zlim(limit(zm, 1), limit(zm, 0))

        # get a list of all lines

        L = list(lc['line' + str(i)] for i in index_l)

        def init():
            for paths in P_list:
                paths.px = []
                paths.py = []
                paths.pz = []
                paths.line[0]._verts3d = (np.array(paths.px), np.array(paths.py), np.array(paths.pz))
            return L

        def update(frame):
            if self.show == 1:
                for paths_ in P_list:
                    paths_.line[0]._marker = matplotlib.lines.MarkerStyle('o')

            for paths in P_list:
                if self.show == 0:
                    paths.px.append(paths.x[frame])
                    paths.py.append(paths.y[frame])
                    paths.pz.append(paths.z[frame])
                    paths.line[0]._verts3d = (np.array(paths.px), np.array(paths.py), np.array(paths.pz))
                elif self.show == 1:
                    paths.line[0]._verts3d = (
                        np.array(paths.x[frame]), np.array(paths.y[frame]), np.array(paths.z[frame]))

            return L

        ani = FuncAnimation(fig, update
                            , frames=convert1(np.linspace(0, len(self.particles[0].x) - 1, len(self.particles[0].x)))
                            , interval=0.1
                            , init_func=init
                            , blit=True
                            )
        plt.show()


if __name__ == '__main__':
    pass
