#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h> // Для работы с приоритетами процессов
#include <errno.h>
#include <elf.h> // Для работы с ELF-файлами
#include <signal.h>

#define PORT 8080     // Порт, на котором работает сервер
#define BUF_SIZE 1024 // Размер буфера для передачи данных

// Функция для вывода списка задач
void list_tasks(int client_fd)
{
    // Запускаем команду "ps" для получения списка задач
    FILE *fp = popen("ps -eo pid,pri,cmd", "r");
    if (fp == NULL)
    {
        perror("popen");
        write(client_fd, "Failed to list tasks\n", 21);
        return;
    }

    char buffer[BUF_SIZE];
    // Читаем и отправляем данные клиенту построчно
    while (fgets(buffer, sizeof(buffer), fp) != NULL)
    {
        write(client_fd, buffer, strlen(buffer));
    }

    pclose(fp);
}

// Функция для завершения задачи по PID
void kill_task(int client_fd, int pid)
{
    if (kill(pid, SIGKILL) == 0)
    { // Отправляем сигнал SIGKILL
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
    { // Устанавливаем новый приоритет
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
    int fd = open(file_path, O_RDONLY); // Открываем файл для чтения
    if (fd < 0)
    {
        perror("open");
        write(client_fd, "Failed to open file\n", 20);
        return;
    }

    Elf64_Ehdr ehdr; // Заголовок ELF-файла
    // Читаем заголовок ELF-файла
    if (read(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr))
    {
        perror("read");
        write(client_fd, "Failed to read ELF header\n", 26);
        close(fd);
        return;
    }

    // Проверяем магическое число ELF-файла
    if (memcmp(ehdr.e_ident, ELFMAG, SELFMAG) != 0)
    {
        write(client_fd, "Not a valid ELF file\n", 22);
        close(fd);
        return;
    }

    // Формируем и отправляем информацию о ELF-файле
    char response[BUF_SIZE];
    snprintf(response, sizeof(response),
             "ELF File Info:\nType: %d\nMachine: %d\nVersion: %d\nEntry point: 0x%lx\n",
             ehdr.e_type, ehdr.e_machine, ehdr.e_version, ehdr.e_entry);
    write(client_fd, response, strlen(response));

    close(fd);
}

// Функция для обработки запросов клиента
void handle_client(int client_fd)
{
    char buffer[BUF_SIZE];

    while (1)
    {
        memset(buffer, 0, BUF_SIZE);                        // Очищаем буфер
        int bytes_read = read(client_fd, buffer, BUF_SIZE); // Читаем данные от клиента
        if (bytes_read <= 0)
        { // Если соединение закрыто, выходим из цикла
            break;
        }

        char command[BUF_SIZE], arg1[BUF_SIZE], arg2[BUF_SIZE];
        int pid, priority;

        // Разбираем команду и её аргументы
        if (sscanf(buffer, "%s %s %s", command, arg1, arg2) >= 1)
        {
            if (strcmp(command, "list") == 0)
            {
                list_tasks(client_fd); // Команда "list"
            }
            else if (strcmp(command, "kill") == 0)
            {
                pid = atoi(arg1);          // Получаем PID
                kill_task(client_fd, pid); // Команда "kill"
            }
            else if (strcmp(command, "priority") == 0)
            {
                pid = atoi(arg1);                          // Получаем PID
                priority = atoi(arg2);                     // Получаем новый приоритет
                change_priority(client_fd, pid, priority); // Команда "priority"
            }
            else if (strcmp(command, "elf") == 0)
            {
                read_elf(client_fd, arg1); // Команда "elf"
            }
            else if (strcmp(command, "exit") == 0)
            {
                break; // Команда "exit"
            }
            else
            {
                write(client_fd, "Unknown command\n", 17); // Неизвестная команда
            }
        }
        else
        {
            write(client_fd, "Invalid input\n", 14); // Некорректный ввод
        }
    }

    close(client_fd); // Закрываем соединение с клиентом
}

int main()
{
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    // Создаём сокет
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1)
    {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    // Задаём параметры адреса сервера
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // Привязываем сокет к адресу
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)
    {
        perror("bind");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // Переводим сокет в режим прослушивания
    if (listen(server_fd, 5) == -1)
    {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("Server is listening on port %d\n", PORT);

    while (1)
    {
        // Принимаем соединение от клиента
        client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd == -1)
        {
            perror("accept");
            continue;
        }

        // Создаём новый процесс для обработки клиента
        if (fork() == 0)
        {
            close(server_fd);         // Закрываем серверный сокет в дочернем процессе
            handle_client(client_fd); // Обрабатываем запросы клиента
            exit(0);
        }

        close(client_fd); // Закрываем клиентский сокет в родительском процессе
    }

    close(server_fd); // Закрываем серверный сокет
    return 0;
}
