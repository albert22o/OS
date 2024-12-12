#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <pthread.h>
#include <sys/resource.h> // Для работы с приоритетами процессов
#include <errno.h>
#include <elf.h> // Для работы с ELF-файлами
#include <signal.h>

#define PORT 8080     // Порт, на котором работает сервер
#define BUF_SIZE 1024 // Размер буфера для передачи данных

// Функция для вывода списка задач и ресурсов
void list(int client_fd)
{
    FILE *fp_tasks = popen("ps -eo pid,pri,%cpu,%mem,cmd", "r");
    if (fp_tasks == NULL)
    {
        perror("popen");
        write(client_fd, "Failed to list tasks and resources\n", 35);
        return;
    }

    char buffer[BUF_SIZE];
    write(client_fd, "PID    PRI   %CPU   %MEM   CMD\n", 31); // Заголовок таблицы

    while (fgets(buffer, sizeof(buffer), fp_tasks) != NULL)
    {
        write(client_fd, buffer, strlen(buffer));
    }

    pclose(fp_tasks);
}

// Функция для завершения задачи по PID
void kill_task(int client_fd, int pid)
{
    if (kill(pid, SIGKILL) == 0)
    {
        char response[BUF_SIZE];
        snprintf(response, sizeof(response), "Task %d terminated successfully\n", pid);
        write(client_fd, response, strlen(response));
    }
    else
    {
        perror("kill");
        write(client_fd, "Failed to terminate task\n", 25);
    }
}

// Функция для изменения приоритета задачи
void change_priority(int client_fd, int pid, int priority)
{
    if (setpriority(PRIO_PROCESS, pid, priority) == 0)
    {
        char response[BUF_SIZE];
        snprintf(response, sizeof(response), "Priority of task %d changed to %d\n", pid, priority);
        write(client_fd, response, strlen(response));
    }
    else
    {
        perror("setpriority");
        write(client_fd, "Failed to change task priority\n", 30);
    }
}

// Функция для чтения информации из ELF-файла
void read_elf(int client_fd, const char *file_path)
{
    int fd = open(file_path, O_RDONLY);
    if (fd < 0)
    {
        perror("open");
        write(client_fd, "Failed to open file\n", 20);
        return;
    }

    Elf64_Ehdr ehdr;
    if (read(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr))
    {
        perror("read");
        write(client_fd, "Failed to read ELF header\n", 26);
        close(fd);
        return;
    }

    if (memcmp(ehdr.e_ident, ELFMAG, SELFMAG) != 0)
    {
        write(client_fd, "Not a valid ELF file\n", 22);
        close(fd);
        return;
    }

    char response[BUF_SIZE];
    snprintf(response, sizeof(response),
             "ELF File Info:\nType: %d\nMachine: %d\nVersion: %d\nEntry point: 0x%lx\n",
             ehdr.e_type, ehdr.e_machine, ehdr.e_version, ehdr.e_entry);
    write(client_fd, response, strlen(response));

    close(fd);
}

// Функция для обработки запросов клиента
void *handle_client(void *arg)
{
    int client_fd = *(int *)arg;
    free(arg);

    char buffer[BUF_SIZE];

    while (1)
    {
        memset(buffer, 0, BUF_SIZE);
        int bytes_read = read(client_fd, buffer, BUF_SIZE);
        if (bytes_read <= 0)
        {
            break;
        }

        char command[BUF_SIZE], arg1[BUF_SIZE], arg2[BUF_SIZE];
        int pid, priority;

        if (sscanf(buffer, "%s %s %s", command, arg1, arg2) >= 1)
        {
            if (strcmp(command, "list") == 0)
            {
                list(client_fd);
            }
            else if (strcmp(command, "kill") == 0)
            {
                pid = atoi(arg1);
                kill_task(client_fd, pid);
            }
            else if (strcmp(command, "priority") == 0)
            {
                pid = atoi(arg1);
                priority = atoi(arg2);
                change_priority(client_fd, pid, priority);
            }
            else if (strcmp(command, "elf") == 0)
            {
                read_elf(client_fd, arg1);
            }
            else if (strcmp(command, "exit") == 0)
            {
                break;
            }
            else
            {
                write(client_fd, "Unknown command\n", 17);
            }
        }
        else
        {
            write(client_fd, "Invalid input\n", 14);
        }
    }

    close(client_fd);
    return NULL;
}

int main()
{
    int server_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1)
    {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)
    {
        perror("bind");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 5) == -1)
    {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("Server is listening on port %d\n", PORT);

    while (1)
    {
        int *client_fd = malloc(sizeof(int));
        if (client_fd == NULL)
        {
            perror("malloc");
            continue;
        }

        *client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (*client_fd == -1)
        {
            perror("accept");
            free(client_fd);
            continue;
        }

        pthread_t thread_id;
        if (pthread_create(&thread_id, NULL, handle_client, client_fd) != 0)
        {
            perror("pthread_create");
            free(client_fd);
            continue;
        }

        pthread_detach(thread_id);
    }

    close(server_fd);
    return 0;
}
