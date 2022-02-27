
from prettytable import PrettyTable
from decimal import Decimal
from math import sqrt
import matplotlib.pyplot as plt


# Подбираем шаг:

# Euler:
# При 1e-1 y(1) = 0.2925421046
# При 1e-2 y(1) = 0.3331073593
# При 1e-3 y(1) = 0.3484859823
# При 1e-4 y(1) = 0.3501691515
# При 1e-5 y(1) = 0.3502255745
# Шаг ничего не меняет (между 1e-3 и 1e-4) 
# Значит мы подобрали нужный нам шаг.

# Runge:
# При 1e-1 y(1) = 0.3485453439
# При 1e-2 y(1) = 0.3391265967
# При 1e-3 y(1) = 0.3491103993
# При 1e-4 y(1) = 0.3502318426
# При 1e-5 y(1) = 0.3502318443
# Аналогично.

MIN_X = -2
MAX_X = 2
STEP = 1e-4

    
def f(x, y):
	return x * x + y * y

def fp1(x):
	return pow(x, 3) / 3

def fp2(x):
	return pow(x, 3) / 3.0 * (1 + pow(x, 4) / 21.0)

def fp3(x):
	return pow(x, 3) / 3.0 * (1.0 +
                            1.0 / 21.0 * pow(x, 4) +
                            2.0 / 693.0 * pow(x, 8) +
                            1.0 / 19845.0 * pow(x, 12))

def fp4(x):
	return pow(x, 3) / 3.0 + pow(x, 7) / 63.0 + pow(x, 11) / 2079.0 * 2.0 +\
            pow(x, 15) / 218295.0 * 13 + pow(x, 19) / 441.0 / 84645.0 * 82.0 +\
            pow(x, 23) / 68607.0 / 152145.0 * 662.0 + pow(x, 27) / pow(3, 11) / 18865.0 * 4.0+\
            pow(x, 31) / 194481.0 / 564975.0

def fp5(x):
	return pow(x, 3) / 3.0 + \
            pow(x, 7) / 63.0 + \
            pow(x, 11) / 2079.0 * 2.0 +\
            pow(x, 15) / 218295.0 * 13 + \
            pow(x, 19) / 654885.0 / 19 * 46 +\
            pow(x, 23) / 1724574159.0 / 23 * 7382 + \
            pow(x, 27) / 1888819317.0 / 27 * 428 +\
            pow(x, 31) / 1686762870563925.0 / 31 * 17843193 + \
            pow(x, 35) / 1725558416586895275.0 / 35 * 738067298 + \
            pow(x, 39) / 688497808218171214725.0 / 39 * 10307579354 + \
            pow(x, 43) / 15530025749282057475.0 / 43 * 6813116 + \
            pow(x, 47) / 8663657814623234993290875.0 / 47 * 89797289962 + \
            pow(x, 51) / 102731120331500810197125.0 / 51 * 19704428 + \
            pow(x, 55) / 278701173339526121443875.0 / 55 * 721012 + \
            pow(x, 59) / 367195221791207011125.0 / 59 * 8 + \
            pow(x, 63) / 12072933807377563850625.0 / 63


def Picar(x_max, h, func):
	result = list()
	x, y = 0, 0

	k = 0
	while x < x_max +h:
		# if (abs(x - k) < 1e-4):
		result.append(y)
			# k = k + 0.05
		x += h
		y = func(x)
	
	return result


def Euler(x_max, h):
	result = list()
	x, y = 0, 0 	# Начальное условие.
	
	k = 0
	while x < x_max +h:
		result.append(y)
		# y[i] = y[i - 1] + step * F(x[i - 1], y[i -1]);
		y = y + h * f(x, y)
		x += h

	return result

def Euler2(x_max, h): # НЕявный
	result = list()
	x, y = 0, 0 	# Начальное условие.
	
	k = 0
	flag = 0
	while x < x_max +h:
		if flag:
			result.append(0)
		else:
			result.append(y)
			x += h
			if ((1.0 / 4.0 / h / h - y / h - x * x) < 0):
				flag = 1
			else:
				y = 1.0 / 2.0 / h - sqrt(1.0 / 4.0 / h / h - y / h - x * x)

	return result


def Runge2(x_max, h):
	result = list()
	coeff = h / 2
	alh = 1
	x, y = 0, 0
	
	while x < x_max +h:
		result.append(y)
		# y[i] = y[i - 1] + step * F(x[i - 1] + coeff, y[i - 1] + coeff * F(x[i - 1], y[i -1]));
		#y = y + h * f(x + coeff, y + coeff * f(x, y))
		y = y + h * ((1 - alh) * f(x, y) + alh * f(x + h / 2 / alh, y + h / 2 / alh * f(x, y)))
		x += h
	
	return result

def Runge4(x_max, h):
	result = list()
	
	x, y = 0, 0
	
	while x < x_max +h:
		result.append(y)
		k1 = h * f(x, y)
		k2 = h * f(x + h / 2, y + k1 / 2)
		k3 = h * f(x + h / 2, y + k2 / 2)
		k4 = h * f(x + h, y + k3)
		# y[i] = y[i - 1] + step * F(x[i - 1] + coeff, y[i - 1] + coeff * F(x[i - 1], y[i -1]));
		#y = y + h * f(x + coeff, y + coeff * f(x, y))
		y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
		x += h
	
	return result



def x_range(x_max, h):
	result = list()
	x = 0
	k = 0
	while x < x_max +h:
		# if (abs(x - k) < 1e-4):
		result.append(x)
			# k = k + 0.05
		x += h
	return result


def main():

	x = x_range(MAX_X, STEP)
	print(len(x))
	p1 = Picar(MAX_X, STEP, fp1)
	print(len(p1))
	p2 = Picar(MAX_X, STEP, fp2)
	p3 = Picar(MAX_X, STEP, fp3)
	p4 = Picar(MAX_X, STEP, fp4)
	p5 = Picar(MAX_X, STEP, fp5)
	ey = Euler(MAX_X, STEP)
	eny = Euler2(MAX_X, STEP)
	runge_k = Runge2(MAX_X, STEP)
	runge_k4 = Runge4(MAX_X, STEP)

	x_res = list()
	p1_res = list()
	p2_res = list()
	p3_res = list()
	p4_res = list()
	p5_res = list()
	ey_res = list()
	eny_res = list()
	runge_res = list()
	runge_res4 = list()

	i = 0
	k = 0
	print(len(x))
	while i < len(x):
		if (abs(k - x[i]) < 1e-4):
			x_res.append(round(x[i], 2))
			p1_res.append(p1[i])
			p2_res.append(p2[i])
			p3_res.append(p3[i])
			p4_res.append(p4[i])
			p5_res.append(p5[i])
			ey_res.append(ey[i])
			eny_res.append(eny[i])
			runge_res.append(runge_k[i])
			runge_res4.append(runge_k4[i])
			k += 0.05
		i += 1

	tb = PrettyTable()
	tb.add_column("X", x_res)
	tb.add_column("Picard 1", p1_res)
	tb.add_column("Picard 2", p2_res)
	tb.add_column("Picard 3", p3_res)
	tb.add_column("Picard 4", p4_res)
	tb.add_column("Picard 5", p5_res)
	tb.add_column("Euler (явный)", ey_res)
	tb.add_column("Euler (неявный)", eny_res)
	tb.add_column("Runge 2", runge_res)
	tb.add_column("Runge 4", runge_res4)

	# tb = PrettyTable()
	# tb.add_column("X", x)
	# tb.add_column("Picard 1", p1)
	# tb.add_column("Picard 2", p2)
	# tb.add_column("Picard 3", p3)
	# tb.add_column("Picard 4", p4)
	# tb.add_column("Picard 5", p5)
	# tb.add_column("Euler (явный)", ey)
	# tb.add_column("Euler (неявный)", eny)
	# tb.add_column("Runge", runge_k)


	print(tb)

	# Запись в файл!
	f = open("./res.txt", "w")
	for i in range(len(x)):
		f.write("%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\n" % (x[i], p1[i], p2[i], p3[i], p4[i], p5[i], ey[i], eny[i], runge_k[i], runge_k4[i]))

	f.close()


	# ГРафик 
	plt.plot(x, runge_k)
	plt.show()


if __name__ == "__main__":
	main()

