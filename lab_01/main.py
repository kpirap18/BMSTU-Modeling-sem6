
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


MAX_X = 2
MIN_X = -MAX_X
STEP = 1e-4
RES_WHAT = 2
    
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


def Picar(x_arr, h, func):
	result = list()

	for x in x_arr:
		y = func(x)
		result.append(y)
	
	return result

def Euler(x_max, h):
	result = list()
	x, y = 0, 0 	# Начальное условие.
	
	k = 0
	while x < x_max + h:
		result.append(y)
		y = y + h * f(x, y)
		x += h

	return result

def Euler_minus(x_min, h):
	h = -h
	result = list()
	x, y = 0, 0 	# Начальное условие.
	
	while x > x_min + h:
		result.append(y)
		y = y + h * f(x, y)
		x += h

	return result

def Euler2(x_max, h): # НЕявный
	result = list()
	x, y = 0, 0 	# Начальное условие.
	
	flag = 0
	while x < x_max + h:
		x += h
		if flag:
			result.append(0)
		else:
			result.append(y)
			if ((1.0 / 4.0 / h / h - y / h - x * x) < 0):
				flag = 1
			else:
				y = 1.0 / 2.0 / h - sqrt(1.0 / 4.0 / h / h - y / h - x * x)


	return result
	
# http://www.mathprofi.ru/metody_eilera_i_runge_kutty.html
def Euler_best(x_max, h): # усовершенсвованный метод Эйлера 
	result = list()
	x, y = 0, 0 	# Начальное условие.
	
	while x < x_max + h:
		result.append(y)
		y = y + h * f(x + h / 2, y + h / 2 * f(x, y))
		x += h

	return result


def Runge2(x_max, h):
	result = list()
	alh = 0.5
	x, y = 0, 0
	
	while x < x_max + h:
		result.append(y)
		y = y + h * ((1 - alh) * f(x, y) + alh * f(x + h / 2 / alh, y + h / 2 / alh * f(x, y)))
		x += h
	
	return result

def Runge2_minus(x_min, h):
	h = -h
	result = list()
	alh = 1
	x, y = 0, 0
	
	while x > x_min + h:
		result.append(y)
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
		y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
		x += h
	
	return result


def Runge5(x_max, h):
	result = list()
	x, y = 0, 0
	
	while x < x_max +h:
		result.append(y)
		k0 = h * f(x, y)
		k1 = h * f(x + h / 3, y + k0 / 3)
		k2 = h * f(x + h / 3, y + k0 / 6 + k1 / 6)
		k3 = h * f(x + h / 2, y + k0 / 8 + 3 * k2 / 8)
		k4 = h * f(x + h, y + k0 / 2 - 3 * k2 / 2 + 2 * k3)
		y = y + (k0 + 4 * k3 + k4) / 6
		x += h
	
	return result

def x_range(x_max, h):
	result = list()
	x = 0
	k = 0
	while x < x_max + h:
		result.append(x)
		x += h
	return result

def x_range_minus(x_min, h):
	h = -h
	result = list()
	x = 0
	k = 0
	while x > x_min + h:
		result.append(x)
		x += h
	return result


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique)) 

	
def main():
	
	x = x_range(MAX_X, STEP)
	print("Массив Х готов!", len(x))
	p1 = Picar(x, STEP, fp1)
	print("1ое приближение", len(p1))
	p2 = Picar(x, STEP, fp2)
	print("2ое приближение", len(p2))
	p3 = Picar(x, STEP, fp3)
	print("3ье приближение", len(p3))
	p4 = Picar(x, STEP, fp4)
	print("4ое приближение", len(p4))
	p5 = Picar(x, STEP, fp5)
	print("5ое приближение", len(p5))
	ey = Euler(MAX_X, STEP)
	print("Эйлер (явно)", len(ey))
	eny = Euler2(MAX_X, STEP)
	print("Эйлер (неявно)", len(eny))
	ey_best = Euler_best(MAX_X, STEP)
	print("Эйлер (улучшенный)", len(ey_best))
	runge_k = Runge2(MAX_X, STEP)
	print("Рунге-Кутта 2 порядок", len(runge_k))
	runge_k4 = Runge4(MAX_X, STEP)
	print("Рунге-Кутта 4 порядок", len(runge_k4))
	runge_k5 = Runge5(MAX_X, STEP)
	print("Рунге-Кутта 5 порядок", len(runge_k5))

	x_res = list()
	p1_res = list()
	p2_res = list()
	p3_res = list()
	p4_res = list()
	p5_res = list()
	ey_res = list()
	eny_res = list()
	ey_best_res = list()
	runge_res = list()
	runge_res4 = list()
	runge_res5 = list()

	i = 0
	k = 0
	while i < len(x):
		if (abs(k - x[i]) < 1e-4):
			x_res.append(round(x[i], 2))
			p1_res.append(round(p1[i], 9))
			p2_res.append(round(p2[i], 9))
			p3_res.append(round(p3[i], 9))
			p4_res.append(round(p4[i], 9))
			p5_res.append(round(p5[i], 9))
			ey_res.append(round(ey[i], 9))
			eny_res.append(round(eny[i], 9))
			ey_best_res.append(round(ey_best[i], 9))
			runge_res.append(round(runge_k[i], 9))
			runge_res4.append(round(runge_k4[i], 9))
			runge_res5.append(round(runge_k5[i], 9))
			k += 0.05
		i += 1


	# Полная или неполная таблица (1 -- не полная, 2 -- полная)
	if (RES_WHAT == 1):
		tb = PrettyTable()
		tb.add_column("X", x_res)
		tb.add_column("Picard 1", p1_res)
		tb.add_column("Picard 2", p2_res)
		tb.add_column("Picard 3", p3_res)
		tb.add_column("Picard 4", p4_res)
		tb.add_column("Picard 5", p5_res)
		tb.add_column("Euler (явный)", ey_res)
		# tb.add_column("Euler (неявный)", eny_res)
		# tb.add_column("Euler (улучшенный)", ey_best_res)
		tb.add_column("Runge 2", runge_res)
		tb.add_column("Runge 4", runge_res4)
		tb.add_column("Runge 5", runge_res5)
	else:
		tb = PrettyTable()
		tb.add_column("X", x)
		tb.add_column("Picard 1", p1)
		tb.add_column("Picard 2", p2)
		tb.add_column("Picard 3", p3)
		tb.add_column("Picard 4", p4)
		tb.add_column("Picard 5", p5)
		tb.add_column("Euler (явный)", ey)
		# tb.add_column("Euler (неявный)", eny)
		# tb.add_column("Euler (улучшенный)", ey_best)
		tb.add_column("Runge", runge_k)
		tb.add_column("Runge 4", runge_k4)
		tb.add_column("Runge 5", runge_k5)

	print(tb)


	# Запись в файл! полные рузультаты
	f = open("./res.txt", "w")
	for i in range(len(x)):
		f.write("%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\n" % (x[i], p1[i], p2[i], p3[i], p4[i], p5[i], ey[i], runge_k[i], runge_k4[i]))

	f.close()


	# ГРафик 
	name_metod = ['Picard 1', 'Picard 2', 'Picard 3', 'Picard 4', 'Euler', 'Runge']
	fig, ax = plt.subplots()
	x_minus = x_range_minus(MIN_X, STEP)
	y_minus = Picar(x_minus, STEP, fp1)
	ax.plot(x, p1, 'r', x_minus, y_minus, 'r', label=name_metod[0])
	y_minus = Picar(x_minus, STEP, fp2)
	ax.plot(x, p2, 'b', x_minus, y_minus, 'b', label=name_metod[1])
	y_minus = Picar(x_minus, STEP, fp3)
	ax.plot(x, p3, 'g', x_minus, y_minus, 'g', label=name_metod[2])
	y_minus = Picar(x_minus, STEP, fp4)
	ax.plot(x, p4, 'y', x_minus, y_minus, 'y', label=name_metod[3])
	y_minus = Euler_minus(MIN_X, STEP)
	ax.plot(x, ey, 'cyan', x_minus, y_minus, 'cyan', label=name_metod[4])
	y_minus = Runge2_minus(MIN_X, STEP)
	ax.plot(x, runge_k, 'purple', x_minus, y_minus, 'purple', label=name_metod[5])

	plt.grid()
	legend_without_duplicate_labels(ax)
	plt.show()


if __name__ == "__main__":
	main()

