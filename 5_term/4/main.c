#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#define NUM_PROC 3

int main()
{
    pid_t pids[NUM_PROC];                                                   // Массив для хранения идентификаторов процессов
    char *programs[NUM_PROC] = {"/bin/ls", "/bin/date", "/usr/bin/whoami"}; // Программы для запуска

    printf("Parent process: PID = %d\n", getppid());
    printf("Executing programs...\n\n");
    for (int i = 0; i < NUM_PROC; i++)
    {
        pids[i] = fork(); // Создаем новый процесс

        if (pids[i] < 0)
        {
            perror("fork failed"); // Проверяем на ошибки
            exit(EXIT_FAILURE);
        }

        if (pids[i] == 0)
        { // В дочернем процессе
            // Выполняем программу
            execlp(programs[i], programs[i], NULL);
            perror("execlp failed"); // Проверяем на ошибки execlp
            exit(EXIT_FAILURE);      // Завершаем в случае ошибки
        }
    }

    // В родительском процессе
    for (int i = 0; i < NUM_PROC; i++)
    {
        int status;
        waitpid(pids[i], &status, 0); // Ждем завершения дочернего процесса
        if (WIFEXITED(status))
        {
            printf("\nChild process %d: PID = %d, PPID = %d\n", i, getpid(), getppid());
            printf("finished with exit status %d\n", WEXITSTATUS(status));
        }
    }

    return 0;
}


    [Родительский процесс]
         /        |        \
   [Процесс 1] [Процесс 2] [Процесс 3]
      (ls)         (date)      (whoami)
