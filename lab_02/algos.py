from cmath import exp
from email.headerregistry import Group
from numpy import e
import matplotlib.pyplot as plt
import seaborn 
import pylab



class Function:
    def __init__(self):
        # константы не изменяемые
        self.k_0 = 0.0008
        self.m = 0.786
        self.R = 0.35
        self.T_w = 2000
        self.T_0 = 10000
        self.c = 3 * 10e10
        self.p = 4
    
    def T(self, z):
        # print("T(Z) = ", (self.T_w - self.T_0) * (z**self.p) + self.T_0)
        return (self.T_w - self.T_0) * (z**self.p) + self.T_0

    def k(self, z):
        return self.k_0 * ((self.T(z) / 300)**2)

    def u_p(self, z):
        # print("Z = ", z)
        return (3.084e-4) / (e**((4.709e+4) / self.T(z)) - 1)

    def U_z(self, z, f):
        return -(3 * self.R * f * self.k(z)) / self.c

    def F_z(self, z, f, u):
        # print("z == 0 ", z,  abs(z - 0) < 1e-4)
        # print("EEEEEEEEEEEEEEEEEEE", f)
        if abs(z - 0) < 1e-4:
            return ((self.R * self.c) / 2) * self.k(z) * (self.u_p(z) - u)
        else:
            return self.R * self.c * self.k(z) * (self.u_p(z) - u) - (f / z)

    def Runge4(self, h0, z0, f0, u0, z_max):
        z_n = z0
        h = h0

        u_n = u0
        f_n = f0

        z_res = [z0]
        u_res = [u0]
        f_res = [f0]

        while z_n < z_max:
            # print("DDDDDD", u_n, f_n)
            k1 = h * self.U_z(z_n, f_n)
            g1 = h * self.F_z(z_n, f_n, u_n)

            k2 = h * self.U_z(z_n + h / 2, f_n + g1 / 2)
            g2 = h * self.F_z(z_n + h / 2, f_n + g1 / 2, u_n + k1 / 2)
            
            k3 = h * self.U_z(z_n + h / 2, f_n + g2 / 2)
            g3 = h * self.F_z(z_n + h / 2, f_n + g2 / 2, u_n + k2 / 2)

            k4 = h * self.U_z(z_n + h, f_n + g3)
            g4 = h * self.F_z(z_n + h, f_n + g3, u_n + k3)

            u_n = u_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            f_n = f_n + (g1 + 2 * g2 + 2 * g3 + g4) / 6

            z_n += h

            z_res.append(z_n)
            u_res.append(u_n)
            f_res.append(f_n)

        print("RESULT", z_res)
        print(u_res)
        print(f_res)
        return z_res, u_res, f_res


class Result(Function):
    def dop_fi(self, f, u):
        res = []
        print(len(f), len(u))
        for i in range(len(u)):
            res.append(f[i] - self.m * self.c * u[i] / 2)

        return res

    def fi(self, f, u):
        # print("FFFFFFFFFF", f, u)
        res = []
        print(len(f), len(u))
        for i in range(len(u)):
            res.append(f[i] - self.m * self.c * u[i] / 2)

        # print(res, res[len(res) - 1])
        return res[len(res) - 1]

    def find_xi(self):
        xi = 0.01
        h = 1e-2
        xi_max = 1

        xi_1 = xi
        xi_2 = xi

        while (xi < xi_max):
            # print("xi", xi)
            z_res, u_res, f_res = self.Runge4(0.01, 0, 0, xi * self.u_p(0), 1)

            fi_1 = self.fi(f_res, u_res)
            # print("fi ", fi_1)

            if (fi_1 < 0):
                xi_2 = xi
                break
            else:
                xi_1 = xi

            xi += h

        return xi_1, xi_2

    def dop_f(self, xi):
        z_res, u_res1, f_res1 = self.Runge4(0.01, 0, 0, xi * self.u_p(0), 1)
        f1 = self.fi(f_res1, u_res1)
        return f1

    def polov_method(self):
        esp = 1e-4

        xi_1, xi_2 = self.find_xi()
        xi = (xi_1 + xi_2) / 2

        t = 0
        while (abs((xi_1 - xi_2) / xi) > esp) and t <101:            
            xi = (xi_1 + xi_2) / 2

            #if (abs(xi_1 - xi_2) < esp):
            #   break

            xi = (xi_1 + xi_2) / 2

            if (self.dop_f(xi_1) * self.dop_f(xi)) >= 0:
                xi_1 = xi
            else:
                xi_2 = xi

            t += 1
            print("RRRRR", (abs((xi_1 - xi_2) / xi) > esp), (abs((xi_1 - xi_2) / xi)), xi_1, xi_2, xi, esp)

        return xi

class Graph(Result):
    def draw(self):
        xi = self.polov_method()
        print("RRRRRRRREWEREWQWERWEWEWE", xi)
        z_res, u_res, f_res = self.Runge4(0.01, 0, 0, xi * self.u_p(0), 1)
        fi_res = self.dop_fi(f_res, u_res)

        name = ['U(z)', 'F(z)']
        # plt.style.use('ggplot')

        plt.subplot(1, 3, 1)
        plt.plot(z_res, u_res, 'r')
        plt.title(name[0])
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.plot(z_res, f_res, 'g')
        plt.title(name[1])
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.plot(z_res, fi_res, 'b')
        plt.title("Result")
        plt.grid()
        
        plt.show()



def main():
    resh = Graph()
    
    # print(resh.find_xi())
    print(resh.draw())

main()
