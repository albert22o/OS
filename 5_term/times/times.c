#include <stdlib.h>
#include <stdio.h>
#include <sys/times.h>
#include <unistd.h>

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
    struct tms start_time, end_time;
    clock_t start, end;

    // начала отсчета
    start = times(&start_time);

    int size = 20 * 1e6;

    double *vector1 = (double *)(calloc(size, sizeof(double)));
    double *vector2 = (double *)(calloc(size, sizeof(double)));

    Init(vector1, size, 0);
    Init(vector2, size, 1);

    InnerProduct(vector1, vector2, size);

    // конец отсчета
    end = times(&end_time);

    // конвертируем время из тиков в секунды
    double total_time = (double)(end - start) / sysconf(_SC_CLK_TCK);

    printf("Время работы программы:\t %f секунд\n", total_time);
    printf("Системное время:\t %f секунд\n", (double)(end_time.tms_stime - start_time.tms_stime) / sysconf(_SC_CLK_TCK));
    printf("Пользовательское время:\t %f секунд\n", (double)(end_time.tms_utime - start_time.tms_utime) / sysconf(_SC_CLK_TCK));

    return 0;
}
