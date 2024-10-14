#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>

#define MAX_PROCESSES 5

// Функция для вывода дерева процессов
void print_process_tree(int level, pid_t pid, const char *name)
{
    for (int i = 0; i < level; i++)
    {
        printf("  "); // Отступ для визуализации уровня
    }
    printf("Process ID: %d, Name: %s\n", pid, name);
}

// Главная функция
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Использование: %s <путь к программе1> <путь к программе2> ...\n", argv[0]);
        return EXIT_FAILURE;
    }

    pid_t pids[MAX_PROCESSES];

    // Запуск процессов
    for (int i = 1; i < argc && i <= MAX_PROCESSES; i++)
    {
        pid_t pid = fork(); // Создаем новый процесс

        if (pid < 0)
        {
            perror("Ошибка fork");
            exit(EXIT_FAILURE);
        }
        else if (pid == 0)
        { // Дочерний процесс
            // Запускаем программу
            execvp(argv[i], &argv[i]); // execvp хэндлит аргументы
            perror("Ошибка execvp");   // Если execvp не удается
            printf("\n");
            exit(EXIT_FAILURE);
        }
        else
        {                      // Родительский процесс
            pids[i - 1] = pid; // Сохраняем идентификатор нового процесса
        }
    }

    // Ждем завершения процессов и выводим дерево процессов
    printf("Дерево процессов:\n");
    for (int i = 0; i < MAX_PROCESSES && pids[i] > 0; i++)
    {
        int status;
        waitpid(pids[i], &status, 0);                // Ждем завершения дочернего процесса
        print_process_tree(1, pids[i], argv[i + 1]); // Печатаем дерево
    }

    return EXIT_SUCCESS;
}
