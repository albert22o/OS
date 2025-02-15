#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#define __RELAXED __ATOMIC_RELAXED
#define __ACQUIRE __ATOMIC_ACQUIRE
#define __RELEASE __ATOMIC_RELEASE
#define __SEQ_CST __ATOMIC_SEQ_CST

volatile int running = 1;
char sh[256];
volatile int flag[2] = {0, 0};
volatile int turn = 0;

void *Thread(void *pParams);

void handle_sigint(int sig)
{
  __atomic_store_n(&running, 0, __RELAXED);
}

int main(void)
{
  pthread_t thread_id;

  signal(SIGINT, handle_sigint);
  pthread_create(&thread_id, NULL, &Thread, NULL);

  while (__atomic_load_n(&running, __SEQ_CST))
  {
    __atomic_store_n(&flag[0], 1, __SEQ_CST);
    __atomic_store_n(&turn, 1, __SEQ_CST);

    while (__atomic_load_n(&flag[1], __SEQ_CST) && __atomic_load_n(&turn, __SEQ_CST) == 1)
    {
    }

    printf("%s", sh);
    fflush(stdout);

    __atomic_store_n(&flag[0], 0, __SEQ_CST);
  }

  pthread_cancel(thread_id);
  pthread_join(thread_id, NULL);

  return 0;
}

void *Thread(void *pParams)
{
  int counter = 0;

  while (__atomic_load_n(&running, __SEQ_CST))
  {
    __atomic_store_n(&flag[1], 1, __SEQ_CST);
    __atomic_store_n(&turn, 0, __SEQ_CST);

    while (__atomic_load_n(&flag[0], __SEQ_CST) && __atomic_load_n(&turn, __SEQ_CST) == 0)
    {
    }

    if (counter > 4)
    {
      __atomic_store_n(&running, 0, __SEQ_CST);
      __atomic_store_n(&flag[1], 0, __SEQ_CST);
      break;
    }

    if (counter % 2)
    {
      strcpy(sh, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    }
    else
    {
      strcpy(sh, "----------------------------------------------------------------------------------------------------------------------------------\n");
    }
    counter++;

    __atomic_store_n(&flag[1], 0, __SEQ_CST);
  }

  return NULL;
}
