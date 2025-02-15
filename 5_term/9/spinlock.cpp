#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <sys/times.h>

#define MAX_COUNT 100000

volatile int running = 1;
int turn = 1;
char sh[6];
pthread_spinlock_t spinlock;

void *Thread(void *pParams);
void handle_sigint(int sig)
{
  running = 0;
}

int main(void)
{
  pthread_t thread_id;
  struct tms start_time, end_time;
  clock_t real_start_time, real_end_time;

  pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);

  signal(SIGINT, handle_sigint);
  pthread_create(&thread_id, NULL, &Thread, NULL);

  real_start_time = times(&start_time);

  while (running)
  {
    pthread_spin_lock(&spinlock);
    if (turn == 0)
    {
      printf("%s", sh);
      fflush(stdout);
      turn = 1;
    }
    pthread_spin_unlock(&spinlock);
  }

  pthread_cancel(thread_id);
  pthread_join(thread_id, NULL);

  pthread_spin_destroy(&spinlock);

  real_end_time = times(&end_time);

  double user_time = (double)(end_time.tms_utime - start_time.tms_utime) / sysconf(_SC_CLK_TCK);
  double system_time = (double)(end_time.tms_stime - start_time.tms_stime) / sysconf(_SC_CLK_TCK);

  printf("User CPU time: %.6f seconds\n", user_time);
  printf("System CPU time: %.6f seconds\n", system_time);

  return 0;
}

void *Thread(void *pParams)
{
  int counter = 0;
  while (counter < MAX_COUNT)
  {
    pthread_spin_lock(&spinlock);
    if (turn == 1)
    {
      if (counter % 2)
      {
        strcpy(sh, "Hello\n");
      }
      else
      {
        strcpy(sh, "Bye_u\n");
      }
      counter++;
      turn = 0;
    }
    pthread_spin_unlock(&spinlock);
  }
  running = 0;
  return NULL;
}
