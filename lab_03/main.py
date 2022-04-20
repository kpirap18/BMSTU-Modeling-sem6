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
        self.k_0 = 0.008 # 0.018
        self.m = 0.786
        self.R = 0.35
        self.T_w = 2000
        self.T_0 = 10000
        self.c = 3e10
        self._p = 4
        self.SHAG_RK = 1e-2
        self.z_max = 1

    
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
        return self.c * self.k(z) * (self.u_p(z) - u)

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
        return (z - self.SHAG_RK / 2) * (self.half_kappa(z - self.SHAG_RK)) / (self.R * self.SHAG_RK) #/ self.R

    def C(self, z):
        return ((z + self.SHAG_RK / 2) * self.half_kappa(z)) / (self.R * self.SHAG_RK) #/ self.R

    def B(self, z):
        return (self.A(z) + self.C(z) + self.p_n(z) * self.V_n(z)) #* z * self.SHAG_RK * self.SHAG_RK * self.R )

    def D(self, z):
        return self.f_n(z) * self.V_n(z) #* z * self.SHAG_RK * self.SHAG_RK * self.R )

    def f_count(self, z, un, un1, f):
        # print(z, un, un1, abs(z - 1) < 1e-4, self.half_kappa(z - self.SHAG_RK))
        if abs(z - 1) < 1e-4:
            return self.m * self.c * un / 2 
        return 2 * self.half_kappa(z - self.SHAG_RK) * (un - un1) / self.SHAG_RK   / self.R

    def f_count2(self, z, z0, u, u0):
        return self.R * (z**2 - z0**2) / z / 2 * (self.f_n(z) + self.f_n(z0) - self.p_n(z) * u - self.p_n(z0) * u0) / 2

    # Краевые условия
    # При х = 0
    def left_boundary_condition(self, z0, F0, h):
        # K0 = self.half_kappa(h / 2, h) * (z0 + h / 2) / (self.R**2 * h) - (h * (self.p_n(0) + self.p_n(h)) * self.V_n(h / 2, h)) / 16
        # M0 = -self.half_kappa(h / 2, h) * (z0 + h / 2) / (self.R**2 * h) - (h * self.p_n(h / 2) * self.V_n(h / 2, h)) / 8 - h * self.p_n(0) * self.V_n(h / 2, h) / 4
        # P0 = -z0 * F0 / self.R - (h * self.V_n(h / 2, h) * (3 * self.f_n(z0) + self.f_n(z0 + h))) / 8
        # K0 = self.half_kappa(h / 2) * (z0 + h / 2) / (self.R**2 * h) - (h * (self.p_n(0) + self.p_n(h)) * self.V_n(h / 2)) / 16
        # M0 = -self.half_kappa(h / 2) * (z0 + h / 2) / (self.R**2 * h) - (h * (self.p_n(0) + self.p_n(h)) * self.V_n(h / 2)) / 16 - h * self.p_n(0) * self.V_n(h / 2) / 4
        # P0 = -z0 * F0 / self.R - (h * self.V_n(h / 2) * (3 * self.f_n(z0) + self.f_n(z0 + h))) / 8
        # print(self.V_n_plus(z0))
        # K0 = self.half_kappa(z0) * (z0 + h / 2) / (self.R**2 * h) - (h * (self.p_n(z0) + self.p_n(z0 + h)) * self.V_n_plus(z0)) / 16
        # M0 = -self.half_kappa(z0) * (z0 + h / 2) / (self.R**2 * h) - (h * (5 * self.p_n(0) + self.p_n(h)) * self.V_n_plus(z0)) / 16
        # P0 = -z0 * F0 / self.R - (h * self.V_n_plus(z0) * (3 * self.f_n(z0) + self.f_n(z0 + h))) / 8
        K0 = -self.half_kappa(z0) * (z0 + h / 2) + self.c * self.R * h * h / 8 * self.k(h / 2) * h / 2
        M0 = self.half_kappa(z0) * (z0 + h / 2) + self.c * self.R * h * h / 8 * self.k(h / 2) * h / 2
        P0 = self.c * self.R * h * h / 4 * self.k(h / 2) * self.u_p(h / 2) * h / 2
        return K0, M0, P0

    # При x = N
    def right_boundary_condition(self, z, h):
        # KN = (h * self.V_n(z - h / 2, h) * self.p_n(z - h / 2)) / 8 - ((z - h / 2) * self.half_kappa(z - h / 2, h)) / (self.R**2 * h) - ((z - h / 2) * self.m * self.c) / self.R / 2
        # MN = -(h * self.V_n(z - h / 2, h) * self.p_n(z - h / 2)) / 8 + (z - h / 2) * self.half_kappa(z - h / 2, h) / self.R**2 / h
        # PN = -(h * self.V_n(h / 2, h) * (self.f_n(z) + self.f_n(z + h / 2))) / 4 
        # KN = -((z - h / 2) * self.half_kappa(z - h / 2)) / (self.R**2 * h) - (h * self.V_n(z - h / 2) * (self.p_n(z - h) + 5 * self.p_n(z))) / 16 - z * self.m * self.c / self.R / 2
        # MN = -(h * self.V_n(z - h / 2) * (self.p_n(z - h) + self.p_n(z))) / 16 + ((z - h / 2) * self.half_kappa(z - h / 2)) / (self.R**2 * h)
        # PN = -(h * self.V_n(z - h / 2) * (3 * self.f_n(z) + self.f_n(z - h))) / 8 
        KN = self.half_kappa(z - h) * (z - h / 2) + self.m * self.c * z * h / 2 + self.c * self.R * h * h / 8 * self.k(z - h / 2) * (z - h / 2)
        MN = -self.half_kappa(z - h) * (z - h / 2) + self.c * self.R * h * h / 8 * self.k(z - h / 2) * (z - h / 2)
        PN = self.c * self.R * h * h / 4 * (self.k(z - h / 2) * self.u_p(z - h / 2) * (z - h / 2) + self.k(z) * self.u_p(z) * z)
        return KN, MN, PN

    def right_hod(self):
        # Прямой ход
        h = self.SHAG_RK
        K0, M0, P0 = self.left_boundary_condition(0, 0, self.SHAG_RK)
        KN, MN, PN = self.right_boundary_condition(1, self.SHAG_RK)  
        # print(K0, M0, P0, KN, MN, PN)      
        eps = [0, -K0 / M0]
        eta = [0, P0 / M0]

        x = h
        n = 1
        while x < 1 + h:
            eps.append(self.C(x) / (self.B(x) - self.A(x) * eps[n]))
            eta.append((self.A(x) * eta[n] + self.D(x)) / (self.B(x) - self.A(x) * eps[n]))
            # eta.append(0)
            n += 1
            x += h
        # eps[n] = 0
        # eta[n] = 1e-6

        # print("EPA\n\n\n",eps, "ETA\n\n\n", eta)
        # Обратный ход
        u = [0] * (n)
        
        u[n-1] = (PN - MN * eta[n]) / (KN + MN * eps[n]) 
        # print("FFFFFFFFFFFFFFFFFF", u[n - 1], n)

        for i in range(n - 2, -1, -1):
            # print(f'i, {u[i]} = {eps[i + 1]} * {u[i + 1]} + {eta[i + 1]}')
            u[i] = eps[i + 1] * u[i + 1] + eta[i + 1]# /8.001

        return u


    def center_formula(self, y, z, h):
        res = []
        res.append((-3 * y[0] + 4 * y[1] - y[2]) / 2 / h)
        for i in range(1, len(y) - 1):
            r = (y[i + 1] - y[i - 1]) / 2 / h
            res.append(r)
        res.append((3 * y[-1] - 4 * y[-2] + y[-3]) / 2 / h)
        print(res, len(res))
        return res


    def F_res_deriv(self, u, z):
        f = [0]
        u_res = self.center_formula(u, z, self.SHAG_RK)
        print(u_res)
        for i in range(1, len(u)):
            r = -self.c / 3 / self.R / self.k(z[i]) * u_res[i]
            f.append(r)
        print("FFFFFFFFF", f)
        return f

        
    def Runge2(self, z_max, h, u_0, u_res):
        result = list()
        alh = 0.5
        x, f = 0, 0
        u = u_0
        
        n = 0
        while x < z_max + h:
            # print(len(u_res) - 1, n, x, z_max + h)
            result.append(f)
            u = u + h * ((1 - alh) * self.U_z(x, f) + alh * self.U_z(x + h / 2 / alh, f + h / 2 / alh * self.U_z(x, f)))
            # print(u, u_res[n])
            f = f + h * ((1 - alh) * self.F_z(x, f, u) + alh * self.F_z(x + h / 2 / alh, f + h / 2 / alh * self.F_z(x, f, u), u + h / 2 / alh * u))
            x += h
            n += 1
        
        return result


    def Euler(self, z_max, h, u_0, u_res):
        result = list()
        x, f = 0, 0 	# Начальное условие.
        u = 2.2224718639841604e-06 #u_0
        
        k = 0
        while x < z_max + h:
            result.append(f)
            u = u + h * self.U_z(x, f)
            # print(u, u_res[k])
            f = f + h * self.F_z(x, f, u)
            x += h
            k += 1

        return result



class Graph(Lab3):
    def draw(self):
        a = Lab3()
        # График
        name = ['U(z)', 'F(z)']
        u_res = a. right_hod()
        z_res = [i for i in np.arange(0, 1 + self.SHAG_RK, a.SHAG_RK)]
        # print(z_res,"\n\n\n U",  u_res)
        #print(u_res, z_res, len(u_res), len(z_res))
        f_res = [0] * len(z_res)
        up_res = [0] * len(z_res)
        divF = [0] * len(z_res)

        print(len(z_res), len(u_res), len(f_res), len(divF))
        f2_res = [0] * len(z_res)
        f3_res = self.F_res_deriv(u_res, z_res)

        for i in range(0, len(u_res) - 1):
            up_res[i] = self.u_p(z_res[i])
            divF[i] = self.divF(z_res[i], u_res[i])
            f2_res[i] = self.f_count2(z_res[i], z_res[0], u_res[i], u_res[0])
        for i in range(1, len(u_res)):
            f_res[i] = self.f_count(z_res[i], u_res[i - 1], u_res[i], f_res[i - 1])
        # f_res[len(z_res) - 1] = self.f_count(z_res[-2], u_res[-1], u_res[len(z_res) - 1], f_res[len(z_res) - 2])

        print(len(z_res), len(u_res), len(f_res), len(divF))
        tb = PrettyTable()
        tb.add_column("Z", z_res)
        tb.add_column("F", f_res)
        tb.add_column("U", u_res)
        tb.add_column("divF", divF)
        # tb.add_column("F (2 порядок)", f3_res)
        # print(tb)

        with open('result.txt', 'w') as f:
            f.write(str(tb))

        # print(z_res[len(z_res) - 1], self.A(z_res[len(z_res) - 1]), self.B(z_res[len(z_res) - 1]), self.C(z_res[len(z_res) - 1]), self.D(z_res[len(z_res) - 1]))

        plt.subplot(2, 2, 1)
        plt.plot(z_res, u_res, 'r', label='u')
        plt.plot(z_res, up_res, 'g', label='u_p')
        plt.legend()
        plt.title(name[0])
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(z_res, f_res, 'g')
        # plt.plot(z_res, f3_res, 'b')
        plt.title(name[1])
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(z_res, divF, 'b')
        plt.title("divF")
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(z_res, f3_res, 'g')
        plt.title("F(z)")
        plt.grid()
        
        plt.show()



def main():
    resh = Graph()
    
    # print(resh.find_xi())
    print(resh.draw())

main()
