#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Для измерения времени

#define ARRAY_SIZE 1000000

int main()
{
  int array[ARRAY_SIZE];
  long long total_sum = 0;
  clock_t start_time, end_time;

  // Инициализация массива
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    array[i] = 1; // Заполнение числами от 1 до ARRAY_SIZE
  }

  // Измерение времени выполнения
  start_time = clock();

  // Вычисление суммы элементов массива
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    total_sum += array[i];
  }

  end_time = clock();

  // Вывод результата и времени выполнения
  printf("Сумма элементов массива: %lld\n", total_sum);
  printf("Время выполнения: %.6f секунд\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

  return 0;
}
