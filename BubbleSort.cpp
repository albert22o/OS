#include <iostream>
#include <ctime>

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

double BubbleSort(int array[], int size)
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
    return clock() / 1000.0;
}

main()
{
    for (int size = 1000; size <= 10000; size += 1000)
    {
        int array[size];
        std::cout << "\nArray size = " << size << std::endl;

        FillRand(array, size);
        std::cout << "Runtime: " << BubbleSort(array, size) << std::endl;
    }
    return 0;
}