#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

int main() {
    pid_t pid = fork(); // Создаем дочерний процесс

    if (pid < 0) {
        perror("fork failed");
        exit(1);
    } else if (pid == 0) {
        // Дочерний процесс
        while (1) {
            printf("Дочерний процесс (PID: %d) выполняется...\n", getpid());
            sleep(1);
        }
    } else {
        // Родительский процесс
        printf("Родительский процесс (PID: %d) создал дочерний процесс (PID: %d)\n", getpid(), pid);
        sleep(20); // Даем дочернему процессу время для выполнения
        kill(pid, SIGTERM); // Завершаем дочерний процесс
    }

    return 0;
}
