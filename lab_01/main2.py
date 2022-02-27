from prettytable import PrettyTable
from math import sqrt

MAX_X = 2
STEP = 1e-5
x_for_table = list()

    
def f(x, y):
	return x * x + y * y
	# return pow(x, 2) + pow(y, 2)

def fp1(x):
	return pow(x, 3) / 3

def fp2(x):
	return pow(x, 7) / 63 + \
		fp1(x)

def fp3(x):
	return pow(x, 15) / 59535 + \
		2 * pow(x, 11) / 2079 + \
		fp2(x)

def fp4(x):
	return pow(x, 31) / 109876903905 + \
		4 * pow(x, 27) / 3341878155 + \
		662 * pow(x, 23) / 10438212015 + \
		82 * pow(x, 19) / 37328445 + \
		fp3(x)

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
	while x < x_max:
		if (abs(x - k) < 1e-4):
			result.append(y)
			k = k + 0.05
		x += h
		y = func(x)
	print("Pic", len(result))
	return result


def Euler(x_max, h): # Явный
	result = list()
	x, y = 0, 0 	# Начальное условие.
	
	k = 0
	while x < x_max:
		if (abs(x - k) < 1e-4):
			result.append(y)
			k = k + 0.05
		# print(y)
		y = y + h * f(x, y)
		x += h

	return result
'''
def Euler2(x_max, h): # НЕявный
	result = list()
	x, y = 0, 0 	# Начальное условие.
	
	k = 0.05
	while x < x_max:
		if (abs(x - k) < 1e-4):
			result.append(y)
			k = k + 0.05
		y = 1.0 / 2.0 / h - sqrt(1.0 / 4.0 / h / h - y / h - x * x)
        x += h
	return result
'''

def Runge(x_max, h):
	result = list()
	coeff = h / 2
	x, y = 0, 0
	
	while x < x_max:
		result.append(y)
		y = y + h * f(x + coeff, y + coeff * f(x, y))
		x += h
	
	return result


def x_range(x_max, h):
	result = list()
	x = 0

	k = 0
	while x < x_max:
		if (abs(x - k) < 1e-4):
			result.append(round(x, 2))
			k = k + 0.05
		x += h
	return result


def main():
	tb = PrettyTable()
	tb.add_column("X", x_range(MAX_X, STEP))
	tb.add_column("Picard 1", Picar(MAX_X, STEP, fp1))
	tb.add_column("Picard 2", Picar(MAX_X, STEP, fp2))
	tb.add_column("Picard 3", Picar(MAX_X, STEP, fp3))
	tb.add_column("Picard 4", Picar(MAX_X, STEP, fp4))
	#tb.add_column("Picard 5", Picar(MAX_X, STEP, fp5))
	tb.add_column("Euler (явный)", Euler(MAX_X, STEP))
	#tb.add_column("Euler (неявный)", Euler2(MAX_X, STEP))
	tb.add_column("Runge", Runge(MAX_X, STEP))

	print(tb)


if __name__ == "__main__":
	main()

