import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns

sns.set_theme(style="darkgrid")

R = 6371302
g = 9.81
U = np.deg2rad(15) / 3600


class BINS:

    def __init__(self, psi, theta, gamma, dpsi,
                 dwbx, dwby, dabx, daby,
                 sigma_a, Tka, sigma_w, Tkw,
                 rand=True, t=91 * 60, dT=.1):
        self.t = t  # время работы
        self.dT = dT  # шаг интегрирования
        self.N = round(self.t / self.dT)
        self.random_proc = rand
        # self.k = число шагов
        self.v = np.array([0, 0, 0])
        self.psi = np.deg2rad(psi)
        self.deltapsi = np.deg2rad(dpsi)
        self.theta = np.deg2rad(theta)
        self.gamma = np.deg2rad(gamma)
        self.phi = np.deg2rad(56)
        self.lambd = np.deg2rad(0)
        self.dab = np.array([dabx * 10 ** -3 * g, daby * 10 ** -3 * g, 0])
        self.dwb = np.array([np.deg2rad(dwbx) / 3600, np.deg2rad(dwby) / 3600, 0])
        self.sigma_a, self.sigma_w = sigma_a * 10 ** -3 * g, np.deg2rad(sigma_w)
        self.Tka, self.Tkw = Tka, Tkw
        self.beta_a, self.beta_w = self.Tka ** -1, self.Tkw ** -1

    @staticmethod
    def get_mnk(gamma, psi, theta):
        c11 = np.cos(gamma) * np.cos(psi) + np.sin(theta) * np.sin(gamma) * np.sin(psi)
        c12 = np.sin(psi) * np.cos(theta)
        c13 = np.cos(psi) * np.sin(gamma) - np.sin(psi) * np.sin(theta) * np.cos(gamma)
        c21 = -np.sin(psi) * np.cos(gamma) + np.cos(psi) * np.sin(theta) * np.sin(gamma)
        c22 = np.cos(psi) * np.cos(theta)
        c23 = -np.sin(psi) * np.sin(gamma) - np.cos(psi) * np.sin(theta) * np.cos(gamma)
        c31 = -np.cos(theta) * np.sin(gamma)
        c32 = np.sin(theta)
        c33 = np.cos(theta) * np.cos(gamma)
        Cbo = np.array([[c11, c12, c13],
                        [c21, c22, c23],
                        [c31, c32, c33]])
        return Cbo

    def ideal_vistavka(self):
        return self.get_mnk(self.gamma, self.psi, self.theta)

    def real_vistavka(self):
        _, mean_a = self.get_measurement("accel")
        theta_align = np.arcsin(mean_a[1] / g)
        gamma_align = -np.arcsin(mean_a[0] / np.sqrt(mean_a[0] ** 2 + mean_a[2] ** 2))
        c_bo_align = self.get_mnk(gamma=gamma_align, psi=self.psi + self.deltapsi, theta=theta_align)
        return c_bo_align

    def get_measurement(self, sensor="accel"):
        if sensor == "accel":
            values = np.array([0, 0, g])
            dv = self.dab
            sigma = self.sigma_a
            beta = self.beta_a
        elif sensor == "gyro":
            values = np.array([0, U * np.cos(self.phi), U * np.sin(self.phi)])
            dv = self.dwb
            sigma = self.sigma_w
            beta = self.beta_w
        else:
            raise RuntimeError("sensor must be accel or gyro")
        Cob = self.ideal_vistavka().T
        a_b = Cob @ values + dv
        ab = np.repeat(a_b[:, np.newaxis], repeats=self.N, axis=1)
        if not self.random_proc:
            return ab, ab.mean(axis=1)
        norm_noise = np.random.normal(size=(3, self.N)) * sigma * np.sqrt(2 * beta * self.dT)
        filtered_noise = np.zeros((3, self.N))
        for i in range(1, self.N):
            filtered_noise[:, i] = (1 - beta * self.dT) * filtered_noise[:, i - 1] + norm_noise[:, i - 1]
        result = ab + filtered_noise
        return result, result.mean(axis=1)

    def navigate_algorythm(self):
        c_bo = self.real_vistavka()
        a_b, _ = self.get_measurement("accel")
        w_b, _ = self.get_measurement("gyro")
        v = self.v
        phi = self.phi
        lambd = self.lambd
        errors = np.array([])
        for i in range(self.N):
            a_o = c_bo * a_b[:, i]
            u_o = self.to_cososym(np.array(0, U * np.cos(phi), U * np.sin(phi)))
            omega_O = self.to_cososym((1 / R) * np.array(-1, 1, np.tan(phi)) * v)
            v = v + self.dT * (a_o + (2 * u_o + omega_O) @ v + np.array([0, 0, -g]))
            w_o = self.to_cososym(
                (1 / R) * np.array(-1, 1, np.tan(phi)) * v + np.array(0, U * np.cos(phi), U * np.sin(phi)))
            w_b_m = self.to_cososym(w_b[:, i])
            c_bo = c_bo + self.dT * (w_o @ c_bo - c_bo @ w_b_m)
            phi += self.dT * v[1] / R
            lambd += self.dT * v[0] / (R * np.cos(phi))
            psi = np.arctan(c_bo[0, 1] / c_bo[1, 1])
            gamma = -np.arctan(c_bo[2, 0] / c_bo[2, 2])
            theta = np.arctan(c_bo[2, 1] / np.sqrt(c_bo[2, 0] ** 2 + c_bo[2, 2] ** 2))
            psi_err, gamma_err, theta_err = psi - self.psi, gamma - self.gamma, theta - self.theta
            np.append(errors, np.array([psi_err, gamma_err, theta_err, lambd, phi, psi, v[0], v[1], v[2]]))
        return errors

    @staticmethod
    def to_cososym(vect):
        return np.array([[0, vect[2], -vect[1]],
                         [-vect[1], 0, vect[0]],
                         [vect[1], -vect[0], 0]])

    @staticmethod
    def plot_acf(data):
        corr = sm.tsa.stattools.acf(data)
        res = np.append(corr[::-1], corr[1:])
        plt.plot(range(-len(res), len(res), 2), res)
        plt.show()


if __name__ == '__main__':
    bins = BINS(psi=90, theta=3, gamma=0,
                dpsi=-0.5, dwbx=1, dwby=0.5,
                dabx=1, daby=0.5, sigma_a=1,
                Tka=0.2, sigma_w=2, Tkw=0.1,
                rand=False)
    # a, means = bins.get_measurement("gyro")
    # bins.plot_acf(a[0])
    # ax = sns.lineplot(a[1])
    # print(f"{means=}")
    # ax.text(1,1,f"{means[0]=}")
    # sns.lineplot(a[1])
    # sns.lineplot(a[2])
    # plt.show()
    print(bins.real_vistavka())
