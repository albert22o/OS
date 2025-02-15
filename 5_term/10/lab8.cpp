#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ARRAY_SIZE 1000000
#define NUM_THREADS 2

typedef struct
{
  int start;
  int end;
  int *array;
  long long partial_sum;
} ThreadData;

volatile int flag[NUM_THREADS];
volatile int turn;

long long total_sum = 0;

void *calculate_partial_sum(void *arg)
{
  ThreadData *data = (ThreadData *)arg;
  data->partial_sum = 0;

  for (int i = data->start; i < data->end; i++)
  {
    data->partial_sum += data->array[i];
  }

  int thread_id = data->start / (ARRAY_SIZE / NUM_THREADS);
  flag[thread_id] = 1;
  turn = thread_id;
  for (int i = 0; i < NUM_THREADS; i++)
  {
    if (i == thread_id)
      continue;
    while (flag[i] && turn == thread_id)
      ;
  }

  total_sum += data->partial_sum;

  flag[thread_id] = 0;

  return NULL;
}

int main()
{
  int array[ARRAY_SIZE];
  pthread_t threads[NUM_THREADS];
  ThreadData thread_data[NUM_THREADS];
  clock_t start_time, end_time;

  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    array[i] = 1;
  }

  for (int i = 0; i < NUM_THREADS; i++)
  {
    flag[i] = 0;
  }
  turn = 0;

  start_time = clock();

  int segment_size = ARRAY_SIZE / NUM_THREADS;
  for (int i = 0; i < NUM_THREADS; i++)
  {
    thread_data[i].start = i * segment_size;
    thread_data[i].end = (i == NUM_THREADS - 1) ? ARRAY_SIZE : (i + 1) * segment_size;
    thread_data[i].array = array;

    pthread_create(&threads[i], NULL, calculate_partial_sum, &thread_data[i]);
  }

  for (int i = 0; i < NUM_THREADS; i++)
  {
    pthread_join(threads[i], NULL);
  }

  end_time = clock();

  printf("Сумма элементов массива: %lld\n", total_sum);
  printf("Время выполнения: %.6f секунд\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

  return 0;
}
