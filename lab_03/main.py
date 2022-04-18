from cmath import exp
from email.headerregistry import Group
from re import L
from numpy import e
import matplotlib.pyplot as plt
import seaborn 
import pylab
from prettytable import PrettyTable
import numpy as np


class Function:
    def __init__(self):
        # константы не изменяемые
        self.k_0 = 0.0008 # 0.018
        self.m = 0.786
        self.R = 0.35
        self.T_w = 2000
        self.T_0 = 10000
        self.c = 3e10
        self._p = 4
        self.SHAG_RK = 1e-2
        self.N = 20
    
    def T(self, z):
        return (self.T_w - self.T_0) * (z**self._p) + self.T_0

    def k(self, z):
        return self.k_0 * ((self.T(z) / 300)**2)

    def u_p(self, z):
        return (3.084e-4) / (e**((4.799e+4) / self.T(z)) - 1)

    def U_z(self, z, f):
        return -(3 * self.R * f * self.k(z)) / self.c

    def F_z(self, z, f, u):
        if abs(z - 0) < 1e-4:
            return ((self.R * self.c) / 2) * self.k(z) * (self.u_p(z) - u)
        else:
            return self.R * self.c * self.k(z) * (self.u_p(z) - u) - (f / z)
        
    
    
    def divF(self, z, u):
        return self.R * self.c * self.k(z) * (self.u_p(z) - u)

class Lab3(Function):
    def k_n(self, z):
        return self.c / (3 * self.R * self.k(z))

    def half_kappa(self, z):
        return (self.k_n(z) + self.k_n(z + self.SHAG_RK)) / 2

    def f_n(self, z):
        return self.c * self.k(z) * self.u_p(z)

    def p_n(self, z):
        return self.c * self.k(z)

    def V_n(self, z):
        return ((z + self.SHAG_RK / 2)**2 - (z - self.SHAG_RK / 2)**2) / 2

    def V_n_plus(self, z):
        return ((z + self.SHAG_RK / 2)**2 - (z)**2) / 2

    def V_n_minus(self, z):
        return ((z)**2 - (z - self.SHAG_RK / 2)**2) / 2


    # # Простая аппроксимация
    def approc_plus_half(self, func, n):
        return (func(n) + func(n + self.SHAG_RK)) / 2

    def approc_minus_half(self, func, n):
        return (func(n - self.SHAG_RK) + func(n)) / 2

    def A(self, z):
        return (z - self.SHAG_RK / 2) * (self.half_kappa(z - self.SHAG_RK)) / (self.R**2 * self.SHAG_RK)

    def C(self, z):
        return ((z + self.SHAG_RK / 2) * self.half_kappa(z)) / (self.R**2 * self.SHAG_RK)

    def B(self, z):
        return self.A(z) + self.C(z) + self.p_n(z) * self.V_n(z)

    def D(self, z):
        return self.f_n(z) * self.V_n(z)

    def f_count(self, z, un, un1):
        print(z, un, un1, abs(z - 1) < 1e-4, self.half_kappa(z))
        if abs(z - 1) < 1e-4:
            return self.m * self.c * un / 2
        return self.half_kappa(z) * (un - un1) / self.SHAG_RK

    # Краевые условия
    # При х = 0
    def left_boundary_condition(self, z0, F0, h):
        # K0 = self.half_kappa(h / 2, h) * (z0 + h / 2) / (self.R**2 * h) - (h * (self.p_n(0) + self.p_n(h)) * self.V_n(h / 2, h)) / 16
        # M0 = -self.half_kappa(h / 2, h) * (z0 + h / 2) / (self.R**2 * h) - (h * self.p_n(h / 2) * self.V_n(h / 2, h)) / 8 - h * self.p_n(0) * self.V_n(h / 2, h) / 4
        # P0 = -z0 * F0 / self.R - (h * self.V_n(h / 2, h) * (3 * self.f_n(z0) + self.f_n(z0 + h))) / 8
        # K0 = self.half_kappa(h / 2) * (z0 + h / 2) / (self.R**2 * h) - (h * (self.p_n(0) + self.p_n(h)) * self.V_n(h / 2)) / 16
        # M0 = -self.half_kappa(h / 2) * (z0 + h / 2) / (self.R**2 * h) - (h * (self.p_n(0) + self.p_n(h)) * self.V_n(h / 2)) / 16 - h * self.p_n(0) * self.V_n(h / 2) / 4
        # P0 = -z0 * F0 / self.R - (h * self.V_n(h / 2) * (3 * self.f_n(z0) + self.f_n(z0 + h))) / 8
        print(self.V_n_plus(z0))
        K0 = self.half_kappa(z0) * (z0 + h / 2) / (self.R**2 * h) - (h * (self.p_n(z0) + self.p_n(z0 + h)) * self.V_n_plus(z0)) / 16
        M0 = -self.half_kappa(z0) * (z0 + h / 2) / (self.R**2 * h) - (h * (5 * self.p_n(0) + self.p_n(h)) * self.V_n_plus(z0)) / 16
        P0 = -z0 * F0 / self.R - (h * self.V_n_plus(z0) * (3 * self.f_n(z0) + self.f_n(z0 + h))) / 8
        return K0, M0, P0

    # При x = N
    def right_boundary_condition(self, z, h):
        # KN = (h * self.V_n(z - h / 2, h) * self.p_n(z - h / 2)) / 8 - ((z - h / 2) * self.half_kappa(z - h / 2, h)) / (self.R**2 * h) - ((z - h / 2) * self.m * self.c) / self.R / 2
        # MN = -(h * self.V_n(z - h / 2, h) * self.p_n(z - h / 2)) / 8 + (z - h / 2) * self.half_kappa(z - h / 2, h) / self.R**2 / h
        # PN = -(h * self.V_n(h / 2, h) * (self.f_n(z) + self.f_n(z + h / 2))) / 4 
        # KN = -((z - h / 2) * self.half_kappa(z - h / 2)) / (self.R**2 * h) - (h * self.V_n(z - h / 2) * (self.p_n(z - h) + 5 * self.p_n(z))) / 16 - z * self.m * self.c / self.R / 2
        # MN = -(h * self.V_n(z - h / 2) * (self.p_n(z - h) + self.p_n(z))) / 16 + ((z - h / 2) * self.half_kappa(z - h / 2)) / (self.R**2 * h)
        # PN = -(h * self.V_n(z - h / 2) * (3 * self.f_n(z) + self.f_n(z - h))) / 8 
        KN = -((z - h / 2) * self.half_kappa(z - h)) / (self.R**2 * h) - (h * self.V_n_minus(z) * (self.p_n(z - h) + 5 * self.p_n(z))) / 16 - z * self.m * self.c / self.R / 2
        MN = -(h * self.V_n_minus(z) * (self.p_n(z - h) + self.p_n(z))) / 16 + ((z - h / 2) * self.half_kappa(z - h)) / (self.R**2 * h)
        PN = -(h * self.V_n_minus(z) * (3 * self.f_n(z) + self.f_n(z - h))) / 8 
        return KN, MN, PN

    def str_hod(self):
        # Прямой ход
        h = self.SHAG_RK
        K0, M0, P0 = self.left_boundary_condition(0, 0, self.SHAG_RK)
        KN, MN, PN = self.right_boundary_condition(1, self.SHAG_RK)  
        print(K0, M0, P0, KN, MN, PN)      
        eps = [0, -K0 / M0]
        eta = [0, P0 / M0]

        x = h
        n = 1
        while x + h < 1 +h:
            eps.append(self.C(x) / (self.B(x) - self.A(x) * eps[n]))
            eta.append((self.A(x) * eta[n] + self.D(x)) / (self.B(x) - self.A(x) * eps[n]))
            # eta.append(0)
            n += 1
            x += h
        # eps[n] = 0
        # eta[n] = 1e-6

        print("EPA\n\n\n",eps, "ETA\n\n\n", eta)
        # Обратный ход
        u = [0] * (n)
        
        u[n-1] = (PN - MN * eta[n]) / (KN + MN * eps[n]) #1.609100e-07 
        print("FFFFFFFFFFFFFFFFFF", u[n - 1], n)

        for i in range(n - 2, -1, -1):
            print(f'i, {u[i]} = {eps[i + 1]} * {u[i + 1]} + {eta[i + 1]}')
            u[i] = eps[i + 1] * u[i + 1] + eta[i + 1]# /8.001

        return u


class Graph(Lab3):
    def draw(self):
        a = Lab3()
        # График
        name = ['U(z)', 'F(z)']
        u_res = a.str_hod()
        z_res = [i for i in np.arange(0, 1, a.SHAG_RK)]
        print(z_res,"\n\n\n U",  u_res)
        #print(u_res, z_res, len(u_res), len(z_res))
        f_res = [self.f_count(0, u_res[0], u_res[1])] * len(z_res)
        up_res = [0] * len(z_res)
        divF = [0] * len(z_res)

        
        for i in range(0, len(u_res) - 1):
            up_res[i] = self.u_p(z_res[i])
            divF[i] = self.divF(z_res[i], u_res[i])
        for i in range(1, len(u_res) - 1):
            f_res[i] = self.f_count(z_res[i], u_res[i], u_res[i + 1])
        f_res[len(u_res) - 1] = self.f_count(1, u_res[-1], 0)

        print(len(z_res), len(u_res), len(f_res), len(divF))
        tb = PrettyTable()
        tb.add_column("Z", z_res)
        tb.add_column("F", f_res)
        tb.add_column("U", u_res)
        tb.add_column("divF", divF)
        print(tb)

        plt.subplot(2, 2, 1)
        plt.plot(z_res, u_res, 'r', label='u')
        plt.plot(z_res, up_res, 'g', label='u_p')
        plt.legend()
        plt.title(name[0])
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(z_res, f_res, 'g')
        # plt.title(name[1])
        plt.grid()

        plt.subplot(2, 2, 3)
        # plt.plot(z_res, fi_res, 'b')
        # st = "XI = " + str(xi)
        # plt.title(st)
        plt.grid()

        plt.subplot(2, 2, 4)
        # plt.plot(z_res, t_res, 'g')
        plt.title("T(z)")
        plt.grid()
        
        plt.show()



def main():
    resh = Graph()
    
    # print(resh.find_xi())
    print(resh.draw())

main()
