#include <iostream>
#include <vector>
#include <thread>
#include <ctime> 

#define ARRAY_SIZE 1000000
#define NUM_THREADS 8

void calculate_partial_sum(const std::vector<int> &array, int start, int end, long long &partial_sum)
{
  partial_sum = 0;
  for (int i = start; i < end; i++)
  {
    partial_sum += array[i];
  }
}

int main()
{
  std::vector<int> array(ARRAY_SIZE);
  long long total_sum = 0;
  std::vector<std::thread> threads(NUM_THREADS);
  std::vector<long long> partial_sums(NUM_THREADS);

  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    array[i] = 1; 
  }

  clock_t start_time = clock();

  int segment_size = ARRAY_SIZE / NUM_THREADS;
  for (int i = 0; i < NUM_THREADS; i++)
  {
    int start = i * segment_size;
    int end = (i == NUM_THREADS - 1) ? ARRAY_SIZE : (i + 1) * segment_size;

    threads[i] = std::thread(calculate_partial_sum, std::cref(array), start, end, std::ref(partial_sums[i]));
  }

  for (int i = 0; i < NUM_THREADS; i++)
  {
    threads[i].join();
  }

  for (const auto &partial_sum : partial_sums)
  {
    total_sum += partial_sum;
  }

  clock_t end_time = clock();
  double duration = double(end_time - start_time) / CLOCKS_PER_SEC;

  std::cout << "Сумма элементов массива: " << total_sum << std::endl;
  std::cout << "Время выполнения: " << duration << " секунд" << std::endl;

  return 0;
}
