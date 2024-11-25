#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

// Глобальная переменная для выхода из цикла
volatile int running = 1;
char sh[6];

void *Thread(void *pParams);

void handle_sigint(int sig)
{
    running = 0; // Устанавливаем флаг выхода при получении сигнала
}

int main(void)
{
    pthread_t thread_id;
    signal(SIGINT, handle_sigint); // Обрабатываем сигнал SIGINT (например, Ctrl+C)
    pthread_create(&thread_id, NULL, &Thread, NULL);

    while (running)
    {
        printf("%s", sh);
        fflush(stdout); // Убедимся, что выводим данные сразу
    }

    pthread_cancel(thread_id);     // Отменяет поток перед выходом
    pthread_join(thread_id, NULL); // Дожидаемся завершения потока

    return 0;
}

void *Thread(void *pParams)
{
    int counter = 0;
    while (running)
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
    }
    return NULL;
}
