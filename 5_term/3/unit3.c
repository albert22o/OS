#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    pid_t pid1, pid2;

    // Создаем первый дочерний процесс
    pid1 = fork();

    if (pid1 < 0) {
        perror("fork failed");
        exit(1);
    } else if (pid1 == 0) {
        // В первом дочернем процессе создаем еще один процесс
        pid2 = fork();

        if (pid2 < 0) {
            perror("fork failed");
            exit(1);
        } else if (pid2 == 0) {
            // Второй дочерний процесс
            printf("Второй дочерний процесс (PID: %d, родитель: %d)\n", getpid(), getppid());
            sleep(60);  // Ждем, чтобы процессы были активны
        } else {
            // Первый дочерний процесс
            printf("Первый дочерний процесс (PID: %d, родитель: %d)\n", getpid(), getppid());
            sleep(60);  // Ждем, чтобы процессы были активны
        }
    } else {
        // Родительский процесс
        printf("Родительский процесс (PID: %d)\n", getpid());
        sleep(60);  // Ждем, чтобы процессы были активны
        exit(0);
    }

    return 0;
}
