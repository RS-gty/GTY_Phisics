from Object import *

def main():
    F = InverseSquareForce(3, [0, 0, 0])
    F2 = InverseSquareForce(5, [0, 0, 0])

    P = Particle()
    P.V_speed = np.array([np.float64(1), np.float64(0), np.float64(0)])
    P.position = np.array([np.float64(0), np.float64(1), np.float64(0)])
    P.related_force = [F2]
    P2 = Particle()
    P2.V_speed = np.array([np.float64(-1), -np.sqrt(0), np.float64(0)])
    P2.position = np.array([np.float64(0), np.float64(-1), np.float64(0)])
    P2.related_force = [F]

    S = Simulation([P2, P], [F, F2])
    S.simulate(20, show_force_source=True)


if __name__ == '__main__':
    main()
