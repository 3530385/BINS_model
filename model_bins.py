import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from tqdm import tqdm
import scipy

sns.set_theme(style="darkgrid")

R = 6371302
g = 9.81
U = np.deg2rad(15) / 3600


class BINS:

    def __init__(self, psi, theta, gamma, dpsi,
                 dwbx, dwby, dabx, daby,
                 sigma_a, Tka, sigma_w, Tkw,
                 rand=True, t=60 * 60, dT=.005):
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
        # print(theta_align, gamma_align)
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
            if (c_bo[0, 1] > 0 > c_bo[1, 1]) or (c_bo[0, 1] < 0 < c_bo[1, 1]):  # Учёт ошибки при выходе из области
                psi = psi + np.deg2rad(180)  # определения арктангенса
            gamma = -np.arctan(c_bo[2, 0] / c_bo[2, 2])
            theta = np.arctan(c_bo[2, 1] / np.sqrt(c_bo[2, 0] ** 2 + c_bo[2, 2] ** 2))
            psi_err, gamma_err, theta_err = psi - self.psi, gamma - self.gamma, theta - self.theta
            current_errors = np.array([np.rad2deg(psi_err) * 60, np.rad2deg(gamma_err) * 60, np.rad2deg(theta_err) * 60,
                                       np.rad2deg(lambd - self.lambd ),
                                       np.rad2deg(phi - self.phi),
                                       np.rad2deg(psi),
                                       v[0], v[1], v[2]])
            errors[:, i] = current_errors
        return errors

    def theory_errors(self):
        # nu = 2 * np.pi / (84.4 * 60)
        nu = np.sqrt(g / R)
        t = np.arange(0, self.t, self.dT)
        c_bo = self.real_vistavka()
        F_ox0 = -1 * self.dab[1] / g
        F_oy0 = self.dab[0] / g
        dao = c_bo @ self.dab
        dwo = c_bo @ self.dwb + np.array([U * np.cos(self.phi) * F_ox0 - U * np.sin(self.phi) * F_oy0,
                                          U * np.sin(self.phi) * F_ox0,
                                          0])  # if not np.allclose(np.array([0,0,0])) else self.get_summ_drift()
        F_ox = -1 * dao[1] / g - dwo[0] * np.sin(nu * t) / nu
        F_oy = dao[0] / g - dwo[1] * np.sin(nu * t) / nu
        dv_ox = dwo[1] * R * (1 - np.cos(nu * t))
        dv_oy = -1 * dwo[0] * R * (1 - np.cos(nu * t))
        d_theta = -1 * F_ox * np.cos(self.psi) + F_oy * np.sin(self.psi)
        d_gamma = - 1 / np.cos(self.theta) * (F_oy * np.cos(self.psi) + F_ox * np.sin(self.psi))
        return dv_ox, dv_oy, np.rad2deg(d_theta) * 60, np.rad2deg(d_gamma) * 60

    def plot_errors(self):
        t = np.arange(0, self.t, self.dT) / 60
        errors = self.navigate_algorythm()
        magic_podgon_alg, magic_shift_alg = np.repeat(np.array([1, 1, 1, 1])[:, np.newaxis],  # w != 0 a != 0
                                                      repeats=self.N, axis=1), \
            np.repeat(np.array([0, 0, 0, 0])[:, np.newaxis],  #
                      repeats=self.N, axis=1)
        magic_podgon_teor, magic_shift_teor = np.repeat(np.array([1, 1, 1, 1])[:, np.newaxis],  #
                                                        repeats=self.N, axis=1), \
            np.repeat(np.array([0, 0, 0, 0])[:, np.newaxis],  #
                      repeats=self.N, axis=1)  #

        # magic_podgon_alg, magic_shift_alg = np.repeat(np.array([1, 1, 1, 1])[:, np.newaxis],           # w == 0 a != 0
        #                                               repeats=self.N, axis=1), \
        #                                     np.repeat(np.array([0, 0, 0, 0])[:, np.newaxis],                         #
        #                                               repeats=self.N, axis=1)                                        #
        #
        # magic_podgon_teor, magic_shift_teor = np.repeat(np.array([3.5, 8, 1.5, 5])[:, np.newaxis],               #
        #                                                 repeats=self.N, axis=1), \
        #                                       np.repeat(np.array([0, 0, -1.9, -34.2+7])[:, np.newaxis],             #
        #                                                 repeats=self.N, axis=1)                                      #

        dv_ox_teor, dv_oy_teor, dtheta_teor, dgamma_teor = self.theory_errors() * magic_podgon_teor + magic_shift_teor
        dv_ox, dv_oy, dtheta, dgamma = np.array([errors[6],
                                                 errors[7],
                                                 errors[2],
                                                 errors[1]]) * magic_podgon_alg + magic_shift_alg
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

        # Показания акселерометров
        self.plot_measurement_for("accel", ab, mean_a, self.sigma_a, t)

        # Показания гироскопов
        self.plot_measurement_for("gyro", wb, mean_w, self.sigma_w, t)

        # Характеристики случайного шума акселерометров
        self.plot_acf_nsd_for("accel", ab[0], self.sigma_a, self.beta_a, self.dT)

        # Характеристики случайного шума гироскопов
        self.plot_acf_nsd_for("gyro", wb[0], self.sigma_w, self.beta_w, self.dT)

    @staticmethod
    def get_y_ticks(mean, std):
        return np.arange(mean - 3 * std, mean + 3 * std, std)

    @staticmethod
    def to_cososym(vect):
        return np.array([[0, vect[2], -vect[1]],
                         [-vect[2], 0, vect[0]],
                         [vect[1], -vect[0], 0]])

    def plot_measurement_for(self, sensor, data, mean, sigma, t):
        if sensor == "accel":
            suptitle = "Показания акселерометров"
            label = ["Осредненное значение $\hat{a} = $" +
                     f"{np.round(mean[i], 3)}" +
                     "$\\frac{м}{с^2}$" for i in range(3)]
            ylabel = [f"$a_{axis}$, м/$с^2$" for axis in ("x", "y", "z")]
        elif sensor == "gyro":
            suptitle = "Показания гироскопов"
            label = ["Осредненное значение $\hat{\omega} = $" +
                     f"{np.round(mean[i], 3)}" +
                     "$\\frac{рад}{с}$" for i in range(3)]
            ylabel = [f"$ \omega _{axis} $, рад/с" for axis in ("x", "y", "z")]
        else:
            raise RuntimeError("sensor must be accel or gyro")
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
        fig.suptitle(suptitle)
        for i, ax in enumerate(axes):
            ax.plot(t, data[i], alpha=0.9)
            ax.plot(t, np.ones_like(t) * mean[i], linewidth=4,
                    label=label[i],
                    color="g")
            ax.set_yticks(self.get_y_ticks(mean[i], sigma))
            ax.set_ylabel(ylabel[i])
            ax.legend(loc="lower right")
        axes[2].set_xlabel("Время моделирования, мин.")
        plt.show()

    @staticmethod
    def plot_acf_nsd_for(sensor, data, sigma, beta, dt):
        if sensor == "accel":
            correct_scale = to_mg = 10 ** 6 / g ** 2
            suptitle = "Характеристики случайного шума акселерометров"
            acf_ylabel = 'Значение корреляционной функции, $mg^2$'
            nsd_ylabel = 'Спектральная плотность, $g^2/Гц$'
        elif sensor == "gyro":
            correct_scale = to_mg = 10 ** 6 / g ** 2
            suptitle = "Характеристики случайного шума гироскопов"
            acf_ylabel = 'Значение корреляционной функции, $\\frac{рад^2}{с}$'
            nsd_ylabel = 'Спектральная плотность, $\\frac{рад^2}{с}/Гц$'
        else:
            raise RuntimeError("sensor must be accel or gyro")

        corr = sm.tsa.stattools.acf(data, fft=True, nlags=round(2 / dt))
        res = np.append(corr[::-1], corr[1:])
        tau = np.arange(-len(res) // 2, len(res) // 2) * dt
        fig, (acf, nsd) = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(suptitle)
        acf.plot(tau, res * sigma ** 2 * correct_scale, linewidth=3)
        acf.plot(tau, sigma ** 2 * np.exp(-beta * np.abs(tau)) * correct_scale,
                 linewidth=2, linestyle='dashed', color="limegreen")
        acf.set_xlabel('Интервал корреляции, с')
        acf.set_ylabel(acf_ylabel)
        acf.legend(["КФ по измерениям", "КФ по формуле"])
        acf.set_title("Корреляционная функция")

        fs = 1 / dt
        # f contains the frequency components
        # S is the PSD
        (f, S) = scipy.signal.welch(data, fs, scaling="spectrum", nperseg=1024)
        f *= np.pi * 2
        nsd.semilogy(f[1:-1], S[1:-1], linewidth=3)
        nsd.plot(f, 2 * sigma ** 2 * beta / (np.pi / 2 * ((f) ** 2 + beta ** 2)),
                 linewidth=2, linestyle='dashed', color="limegreen")
        # plt.xlim([0, fs*np.pi*2//2])
        # plt.ylim([2e-8, 4e-5])
        nsd.set_xlabel('Частота, Гц')
        nsd.set_ylabel(nsd_ylabel)
        nsd.set_title("Спектральная плотность сигнала")
        nsd.legend(["СП по смоделированным\n измерениям", "СП по формуле"])
        plt.show()


if __name__ == '__main__':
    bins = BINS(psi=90, theta=-3, gamma=2,
                dpsi=1, dwbx=-2, dwby=1,
                # dpsi=1, dwbx=0, dwby=0,
                # dabx=-2, daby=1, sigma_a=0.5,
                dabx=0, daby=0, sigma_a=0.5,
                Tka=0.2, sigma_w=.05, Tkw=0.1,
                rand=False, t=95 * 60, dT=.1)
    bins.plot_errors()
    # bins = BINS(psi=90, theta=-3, gamma=2,
    #             dpsi=1, dwbx=-2, dwby=1,
    #             # dpsi=1, dwbx=0, dwby=0,
    #             dabx=-2, daby=1, sigma_a=0.5,
    #             # dabx=0, daby=0, sigma_a=0.5,
    #             Tka=0.2, sigma_w=.05, Tkw=0.1,
    #             rand=True, t=5 * 60, dT=.005)
    # bins.plot_measurement()
    # bins.plot_errors()
    # print(bins.ideal_vistavka())
    # print(bins.real_vistavka())