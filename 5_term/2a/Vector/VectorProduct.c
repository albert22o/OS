#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*
flag = 0 -- fill with decreasing numbers
flag = 1 -- fill with increasing numbers
*/
void Init(double *vector, int size, int flag)
{
    switch (flag)
    {
    case 0:
        for (int i = 0; i < size; i++)
            vector[i] = size - i;
    case 1:
        for (int i = 0; i < size; i++)
            vector[i] = size;
    default:
        for (int i = 0; i < size; i++)
            vector[i] = size;
    }
}

double InnerProduct(double *a, double *b, int size)
{
    double result;
    for (int i = 0; i < size; i++)
        result += a[i] * b[i];

    return result;
}

int main()
{
    clock_t start, end;
    double cpu_time_used;
    int size = 20 * 1e6;

    double *vector1 = (double *)(calloc(size, sizeof(double)));
    double *vector2 = (double *)(calloc(size, sizeof(double)));

    start = clock(); // Начало отсчета времени

    Init(vector1, size, 0);
    Init(vector2, size, 1);

    InnerProduct(vector1, vector2, size);

    end = clock(); // Конец отсчета времени
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("SIZE = %d\tRuntime = %f\n", size, cpu_time_used);

    return 0;
}