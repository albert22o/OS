#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void FillRand(int array[], int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        array[i] = rand() % 100;
    }
}

void PrintArr(int array[], int size)
{
    for (int i = 0; i < size; i++)
    {
        printf(" %d", array[i]);
    }
    printf("\n");
}

void BubbleSort(int array[], int size)
{
    int i, j, temp;
    for (i = 0; i < size - 1; i++)
    {
        for (j = size - 1; j >= i + 1; j--)
        {
            if (array[j] < array[j - 1])
            {
                temp = array[j];
                array[j] = array[j - 1];
                array[j - 1] = temp;
            }
        }
    }
}

int main()
{
    clock_t start, end;
    double cpu_time_used;

    for (int size = 1000; size <= 25000; size += 1000)
    {
        int array[size];
        printf("\nArray size = %d", size);

        FillRand(array, size);

        start = clock(); // Начало отсчета времени
        BubbleSort(array, size);
        end = clock(); // Конец отсчета времени
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf("\tRuntime = %.3f", cpu_time_used);
    }

    return 0;
}
