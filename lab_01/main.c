#include <stdio.h>
#include <math.h>

#define N 30000

long double F( long double x, long double u) 
{
    return x*x + u*u;
}

long double pikar1(long double x) 
{
    return pow(x, 3) / 3.0;
}

long double pikar2(long double x) 
{
    return pow(x, 3) / 3.0 * (1 + pow(x, 4) / 21.0);
}

long double pikar3(long double x) 
{
    return pow(x, 3) / 3.0 * (1.0 +
                            1.0 / 21.0 * pow(x, 4) +
                            2.0 / 693.0 * pow(x, 8) +
                            1.0 / 19845.0 * pow(x, 12));
}

long double pikar4(long double x) 
{
    return (pow(x, 3) / 3.0 + pow(x, 7) / 63.0 + pow(x, 11) / 2079.0 * 2.0 +
            pow(x, 15) / 218295.0 * 13 + pow(x, 19) / 441.0 / 84645.0 * 82.0 +
            pow(x, 23) / 68607.0 / 152145.0 * 662.0 + pow(x, 27) / pow(3, 11) / 18865.0 * 4.0+
            pow(x, 31) / 194481.0 / 564975.0);
}

long double pikar5(long double x) 
{
    return (pow(x, 3) / 3.0 + 
            pow(x, 7) / 63.0 + 
            pow(x, 11) / 2079.0 * 2.0 +
            pow(x, 15) / 218295.0 * 13 + 
            pow(x, 19) / 654885.0 / 19 * 46 +
            pow(x, 23) / 1724574159.0 / 23 * 7382 + 
            pow(x, 27) / 1888819317.0 / 27 * 428 +
            pow(x, 31) / 1686762870563925.0 / 31 * 17843193 + 
            pow(x, 35) / 1725558416586895275.0 / 35 * 738067298 + 
            pow(x, 39) / 688497808218171214725.0 / 39 * 10307579354 + 
            pow(x, 43) / 15530025749282057475.0 / 43 * 6813116 + 
            pow(x, 47) / 8663657814623234993290875.0 / 47 * 89797289962 + 
            pow(x, 51) / 102731120331500810197125.0 / 51 * 19704428 + 
            pow(x, 55) / 278701173339526121443875.0 / 55 * 721012 + 
            pow(x, 59) / 367195221791207011125.0 / 59 * 8 + 
            pow(x, 63) / 12072933807377563850625.0 / 63);
}

// явный метод Эйлера
void yavnii(long double *y, long double *x, int lenx, long double step)
{
    y[0] = 0;

    for (int i = 1; i < lenx; i++)
    {
        y[i] = y[i - 1] + step * F(x[i - 1], y[i -1]);
    }
}

// НЕявный метод Эйлера
void neyavnii(long double *y, long double *x, int lenx, long double step)
{
    y[0] = 0;

    for (int i = 1; i < lenx; i++)
    {
        y[i] = 1.0 / 2.0 / step - 
                sqrt(1.0 / 4.0 / step / step - y[i-1] / step - x[i] * x[i]);
    }
}   

// Рунте - Кутта
void runge(long double *y, long double *x, int lenx, long double step)
{
    long double coeff = step / 2;
    y[0] = 0;

    for (int i = 1; i < lenx; i++)
    {
        y[i] = y[i - 1] + step * F(x[i - 1] + coeff, y[i - 1] + coeff * F(x[i - 1], y[i -1]));
    }
}


int main()
{
    setbuf(stdout, NULL);
    printf(" %Lf %Lf %Lf %Lf %Lf", pikar1(2), pikar2(2), pikar3(2), pikar4(2), pikar5(2));


    long double p1[N];
    long double p2[N];
    long double p3[N];
    long double p4[N];
    long double p5[N];
    long double yab[N]; // явный Эйлер
    long double neyab[N]; // НЕявный Эйлер
    long double runge_k[N]; 

    long double x[N];

    float x_start = 0;
    float x_end = 2;
    float x_step = 1e-5;

    float k = x_start;
    for (int i = 0; k < x_end + x_step; i++)
    {
        printf("%d %f\n", i, k);
        x[i] = k;
        k = k + x_step;
    }

    // for (int i = 0; i < N; i++)
    // {
    //     p1[i] = pikar1(x[i]);
    //     p2[i] = pikar2(x[i]);
    //     p3[i] = pikar3(x[i]);
    //     p4[i] = pikar4(x[i]);
    //     p5[i] = pikar5(x[i]);
    // }

    yavnii(yab, x, N, x_step);
    neyavnii(neyab, x, N, x_step);

    runge(runge_k, x, N, x_step);


    printf("|%8s|%10s|%10s|%10s|%10s|%10s|%10s|%10s|%10s|\n", "n", "pikar1", "pikar2", "pikar3", "pikar4", "pikar5", "yavnii", "neyavnii", "Runge");

    k = 0;
    for (int i = 0; i < N; i++)
    {
        if (fabs(x[i] - k) < 1e-4)
        {
            printf("|%8Lf|%10Lf|%10Lf|%10Lf|%10Lf|%10Lf|%10Lf|%10Lf|%10Lf|\n", x[i], p1[i], p2[i], p3[i], p4[i], p5[i], yab[i], neyab[i], runge_k[i]);
            k = k + 0.05;
        }
    }

    return 0;
}