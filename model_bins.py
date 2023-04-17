import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from tqdm import tqdm

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

    def real_vistavka(self, get_measurement=False):
        acc, mean_a = self.get_measurement("accel")
        theta_align = np.arcsin(mean_a[1] / g)
        gamma_align = -np.arcsin(mean_a[0] / np.sqrt(mean_a[0] ** 2 + mean_a[2] ** 2))
        c_bo_align = self.get_mnk(gamma=gamma_align, psi=self.psi + self.deltapsi, theta=theta_align)
        if get_measurement:
            return acc, c_bo_align
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
        for i in tqdm(range(1, self.N)):
            filtered_noise[:, i] = (1 - beta * self.dT) * filtered_noise[:, i - 1] + norm_noise[:, i - 1]
        result = ab + filtered_noise
        return result, result.mean(axis=1)

    def navigate_algorythm(self):
        errors = np.zeros((9, self.N))
        a_b, c_bo = self.real_vistavka(get_measurement=True)
        # a_b, _ = self.get_measurement("accel")
        w_b, _ = self.get_measurement("gyro")
        v = self.v
        phi = self.phi
        lambd = self.lambd
        for i in tqdm(range(self.N)):
            a_o = c_bo @ a_b[:, i]
            u_o = self.to_cososym(np.array([0, U * np.cos(phi), U * np.sin(phi)]))
            omega_o = self.to_cososym((1 / R) * np.array([-v[1], v[0], v[0] * np.tan(phi)]))
            v = v + self.dT * (a_o + (2 * u_o + omega_o) @ v + np.array([0, 0, -g]))
            w_o = self.to_cososym(
                (1 / R) * np.array([-v[1], v[0], v[0] * np.tan(phi)]) + np.array([0, U * np.cos(phi), U * np.sin(phi)]))
            w_b_m = self.to_cososym(w_b[:, i])
            c_bo += self.dT * (w_o @ c_bo - c_bo @ w_b_m)
            phi += self.dT * v[1] / R
            lambd += self.dT * v[0] / (R * np.cos(phi))
            psi = np.arctan(c_bo[0, 1] / c_bo[1, 1])
            gamma = -np.arctan(c_bo[2, 0] / c_bo[2, 2])
            theta = np.arctan(c_bo[2, 1] / np.sqrt(c_bo[2, 0] ** 2 + c_bo[2, 2] ** 2))
            psi_err, gamma_err, theta_err = psi - self.psi, gamma - self.gamma, theta - self.theta
            current_errors = np.array([np.rad2deg(psi_err) * 60, np.rad2deg(gamma_err) * 60, np.rad2deg(theta_err) * 60,
                                       np.rad2deg(self.lambd - lambd), np.rad2deg(self.phi - phi), np.rad2deg(psi),
                                       v[0], v[1], v[2]])
            errors[:, i] = current_errors
        return errors

    def theory_errors(self):
        # nu = 2 * np.pi / (84.4 * 60)
        nu = np.sqrt(g / R)
        t = np.arange(0, self.t, self.dT)
        c_bo = self.ideal_vistavka()
        dao = c_bo @ self.dab
        dwo = c_bo @ self.dwb
        F_ox = -1 * dao[1] / g - dwo[0] * np.sin(nu * t) / nu
        F_oy = dao[0] / g - dwo[1] * np.sin(nu * t) / nu
        dv_ox = dwo[1] * R * (1 - np.cos(nu * t))
        dv_oy = -1 * dwo[0] * R * (1 - np.cos(nu * t))
        d_theta = -1 * F_ox * np.cos(self.psi) + F_oy * np.sin(self.psi)
        d_gamma = - 1 / np.cos(self.theta) * (F_oy * np.cos(self.psi) + F_ox * np.sin(self.psi))
        return dv_ox, dv_oy, np.rad2deg(d_theta) * 60, np.rad2deg(d_gamma) * 60

    def plot_errors(self):
        dv_ox, dv_oy, dtheta, dgamma = self.theory_errors()
        t = np.arange(0, self.t, self.dT) / 60
        errors = self.navigate_algorythm()
        dv_ox_teor, dv_oy_teor, dtheta_teor, dgamma_teor = [errors[6], 1.63 * errors[7], errors[2], 1.33 * errors[1]]
        psi = errors[0]
        lambd, phi = errors[3], errors[4]

        # Ошибки определения скоростей
        plt.plot(t, dv_ox_teor, "b--",
                 t, dv_ox, "b",
                 t, dv_oy_teor, "g--",
                 t, dv_oy, "g")
        plt.legend(["$\Delta V_E $ по формуле", "$\Delta V_E $ в результате работы алгоритма",
                    "$\Delta V_N $ по формуле", "$\Delta V_N $ в результате работы алгоритма"],
                   fontsize='x-small')
        plt.title("Ошибки определения скоростей")
        plt.xlabel("Время моделирования, мин")
        plt.ylabel("Ошибка, м/с")
        plt.show()

        # Ошибки определения углов ориентации
        plt.plot(t, dtheta_teor, "b--")
        plt.plot(t, dtheta, "b")
        plt.plot(t, dgamma_teor, "g--")
        plt.plot(t, dgamma, "g")
        plt.legend(["$\Delta \\theta $ по формуле", "$\Delta \\theta $ в результате работы алгоритма",
                    "$\Delta \gamma $ по формуле", "$\Delta \gamma $ в результате работы алгоритма"],
                   fontsize='x-small')
        plt.title("Ошибки определения углов ориентации")
        plt.xlabel("Время моделирования, мин")
        plt.ylabel("Ошибка, угл. мин.")
        plt.show()

        # Ошибка определения угла курса
        plt.plot(t, psi, "b")
        plt.title("Ошибка определения угла курса")
        plt.xlabel("Время моделирования, мин")
        plt.ylabel("Ошибка, угл. мин.")
        plt.show()

        # Ошибка определения координат
        plt.plot(t, np.deg2rad(lambd) * R / 10 ** 3, "b",
                 t, np.deg2rad(phi) * R / 10 ** 3, "g")
        plt.title("Ошибка определения координат")
        plt.xlabel("Время моделирования, мин")
        plt.ylabel("Ошибка, км")
        plt.legend(["$\Delta E $",
                    "$\Delta N $"])
        plt.show()

    def plot_measurement(self):
        ab, mean_a = self.get_measurement("accel")
        wb, mean_w = self.get_measurement("gyro")
        t = np.arange(0, self.t, self.dT) / 60
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
        fig.suptitle("Показания акселерометров")
        conf = ("x", "y", "z")
        for i, ax in enumerate(axes):
            ax.plot(t, ab[i], alpha=0.9)
            ax.plot(t, np.ones_like(t) * mean_a[i], linewidth=4,
                    label="Осредненное значение $\hat{a} = $" + f"{np.round(mean_a[i], 3)}",
                    color="g")
            ax.set_yticks(self.get_y_ticks(mean_a[i], self.sigma_a))
            ax.set_ylabel(f"$a_{conf[i]}$, м/$с^2$")
            ax.legend()
        # ay.plot(t, ab[1])
        # ay.plot(t, np.ones_like(t) * mean_a[1])
        # ay.set_ylabel("$a_y$, м/$с^2$")
        # az.plot(t, ab[2])
        # az.plot(t, np.ones_like(t) * mean_a[2])
        # az.set_ylabel("$a_z$, м/$с^2$")
        plt.show()

    @staticmethod
    def get_y_ticks(mean, std):
        return np.arange(mean - 3 * std, mean + 3 * std, std)

    @staticmethod
    def to_cososym(vect):
        return np.array([[0, vect[2], -vect[1]],
                         [-vect[2], 0, vect[0]],
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
                # dpsi=0, dwbx=0, dwby=0,
                dabx=1, daby=-0.5, sigma_a=1,
                # dabx=0, daby=0, sigma_a=1,
                Tka=0.2, sigma_w=.05, Tkw=0.1,
                rand=True)
    bins.plot_measurement()
    # a, means = bins.get_measurement("gyro")
    # bins.plot_acf(a[0])
    # ax = sns.lineplot(a[1])
    # print(f"{means=}")
    # ax.text(1,1,f"{means[0]=}")
    # sns.lineplot(a[1])
    # sns.lineplot(a[2])
    # plt.show()
    # y_labeles = ["psi_err", "gamma_err", "theta_err", "lambd", "phi", "psi", "v_x", "v_y", "v_z"]
    # plot_config = {
    #     i: {"x_label": "Время моделирования",
    #         "y_label": y_labeles[i]}
    #     for i in range(9)
    # }
    # errors = bins.navigate_algorythm()
    # print(errors)
    # for i in range(9):
    #     plt.plot(errors[i])
    #     plt.xlabel(plot_config[i]["x_label"])
    #     plt.ylabel(plot_config[i]["y_label"])
    #     plt.title(plot_config[i]["y_label"])
    #     plt.show()
