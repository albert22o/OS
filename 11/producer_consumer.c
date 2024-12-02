#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <signal.h>

#define MAX_ITEMS 8   // Максимальное количество элементов для обработки
#define BUFFER_SIZE 5 // Размер буфера

int buffer[BUFFER_SIZE]; // Буфер
int in = 0, out = 0;     // Индексы для записи и чтения
int produced_count = 0;  // Счётчик произведённых элементов
int consumed_count = 0;  // Счётчик потреблённых элементов
int stop_flag = 0;       // Флаг завершения программы

pthread_mutex_t mutex; // Мьютекс для синхронизации доступа к буферу
sem_t empty;           // Семафор для отслеживания свободных мест
sem_t full;            // Семафор для отслеживания заполненных мест

// Обработчик сигнала завершения (Ctrl+C)
void handle_signal(int sig)
{
    stop_flag = 1; // Устанавливаем флаг завершения
    printf("\nПолучен сигнал завершения. Завершаем программу...\n");
}

// Функция производителя
void *producer(void *arg)
{
    int item;
    while (!stop_flag)
    {
        item = rand() % 100;        // Производим случайный элемент
        sem_wait(&empty);           // Ждём, пока будет свободное место
        pthread_mutex_lock(&mutex); // Входим в критическую секцию

        if (stop_flag)
        { // Проверяем флаг завершения внутри критической секции
            pthread_mutex_unlock(&mutex);
            sem_post(&empty);
            break;
        }

        buffer[in] = item; // Помещаем элемент в буфер
        printf("Производитель произвел: %d\n", item);
        in = (in + 1) % BUFFER_SIZE; // Обновляем индекс
        produced_count++;

        pthread_mutex_unlock(&mutex); // Выходим из критической секции
        sem_post(&full);              // Увеличиваем количество заполненных мест

        if (produced_count >= MAX_ITEMS)
        {
            stop_flag = 1; // Завершаем после достижения лимита
            break;
        }
        sleep(rand() % 2); // Задержка для реалистичности
    }
    return NULL;
}

// Функция потребителя
void *consumer(void *arg)
{
    int item;
    while (!stop_flag || consumed_count < produced_count)
    {
        sem_wait(&full);            // Ждём, пока буфер не станет пустым
        pthread_mutex_lock(&mutex); // Входим в критическую секцию

        if (stop_flag && consumed_count >= produced_count)
        { // Проверяем флаг завершения
            pthread_mutex_unlock(&mutex);
            sem_post(&full);
            break;
        }

        item = buffer[out]; // Извлекаем элемент из буфера
        printf("Потребитель потребил: %d\n", item);
        out = (out + 1) % BUFFER_SIZE; // Обновляем индекс
        consumed_count++;

        pthread_mutex_unlock(&mutex); // Выходим из критической секции
        sem_post(&empty);             // Увеличиваем количество свободных мест

        sleep(rand() % 3); // Задержка для реалистичности
    }
    return NULL;
}

int main()
{
    pthread_t prod, cons;

    // Устанавливаем обработчик сигнала завершения
    signal(SIGINT, handle_signal);

    // Инициализация мьютекса и семафоров
    pthread_mutex_init(&mutex, NULL);
    sem_init(&empty, 0, BUFFER_SIZE);
    sem_init(&full, 0, 0);

    // Создание потоков производителя и потребителя
    pthread_create(&prod, NULL, producer, NULL);
    pthread_create(&cons, NULL, consumer, NULL);

    // Ожидание завершения потоков
    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    // Очистка ресурсов
    pthread_mutex_destroy(&mutex);
    sem_destroy(&empty);
    sem_destroy(&full);

    printf("Программа завершена. Произведено: %d, Потреблено: %d\n", produced_count, consumed_count);
    return 0;
}
