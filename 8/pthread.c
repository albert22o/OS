#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h> // Для измерения времени

#define ARRAY_SIZE 1000000
#define NUM_THREADS 8

// Структура для передачи данных в поток
typedef struct
{
  int start;
  int end;
  int *array;
  long long partial_sum;
} ThreadData;

void *calculate_partial_sum(void *arg)
{
  ThreadData *data = (ThreadData *)arg;
  data->partial_sum = 0;
  for (int i = data->start; i < data->end; i++)
  {
    data->partial_sum += data->array[i];
  }
  return NULL;
}

int main()
{
  int array[ARRAY_SIZE];
  pthread_t threads[NUM_THREADS];
  ThreadData thread_data[NUM_THREADS];
  long long total_sum = 0;
  clock_t start_time, end_time;

  // Инициализация массива
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    array[i] = 1; // Заполнение числами от 1 до ARRAY_SIZE
  }

  // Измерение времени выполнения
  start_time = clock();

  // Разбиение массива на сегменты для потоков
  int segment_size = ARRAY_SIZE / NUM_THREADS;
  for (int i = 0; i < NUM_THREADS; i++)
  {
    thread_data[i].start = i * segment_size;
    thread_data[i].end = (i == NUM_THREADS - 1) ? ARRAY_SIZE : (i + 1) * segment_size;
    thread_data[i].array = array;

    pthread_create(&threads[i], NULL, calculate_partial_sum, &thread_data[i]);
  }

  // Ожидание завершения потоков и суммирование результатов
  for (int i = 0; i < NUM_THREADS; i++)
  {
    pthread_join(threads[i], NULL);
    total_sum += thread_data[i].partial_sum;
  }

  end_time = clock();

  // Вывод результата и времени выполнения
  printf("Сумма элементов массива: %lld\n", total_sum);
  printf("Время выполнения: %.6f секунд\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

  return 0;
}
