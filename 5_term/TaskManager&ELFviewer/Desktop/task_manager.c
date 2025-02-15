#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <ctype.h>
#include <sys/resource.h>
#include <unistd.h>
#include "task_manager.h"

void list_processes()
{
    DIR *proc = opendir("/proc");
    struct dirent *entry;

    if (!proc)
    {
        perror("opendir");
        return;
    }

    printf("PID\tPPID\tMemory\tState\tPriority\tName\n");
    printf("-----------------------------------------------------------\n");

    while ((entry = readdir(proc)) != NULL)
    {
        if (entry->d_type == DT_DIR && isdigit(entry->d_name[0]))
        {
            char path[256], name[256], state;
            int pid, ppid, priority;
            unsigned long mem;
            FILE *stat;

            if (snprintf(path, sizeof(path), "/proc/%s/stat", entry->d_name) >= sizeof(path))
            {
                fprintf(stderr, "Path is too long, skipping: %s\n", entry->d_name);
                continue;
            }
            stat = fopen(path, "r");

            if (stat)
            {
                fscanf(stat, "%d %s %c %d", &pid, name, &state, &ppid);
                fseek(stat, 0, SEEK_SET);
                for (int i = 0; i < 18; i++)
                    fscanf(stat, "%*s");
                fscanf(stat, "%d", &priority);
                for (int i = 0; i < 10; i++)
                    fscanf(stat, "%*s");
                fscanf(stat, "%lu", &mem);
                fclose(stat);
                printf("%d\t%d\t%lu\t%c\t%d\t%s\n", pid, ppid, mem, state, priority, name);
            }
        }
    }

    closedir(proc);
}

void kill_process()
{
    int pid;
    printf("Enter PID to kill: ");
    scanf("%d", &pid);
    if (kill(pid, SIGKILL) == 0)
        printf("Process %d killed.\n", pid);
    else
        perror("kill");
}

void change_priority()
{
    int pid, priority;
    printf("Enter PID to change priority: ");
    scanf("%d", &pid);
    printf("Enter new priority (-20 to 19): ");
    scanf("%d", &priority);

    if (setpriority(PRIO_PROCESS, pid, priority) == 0)
        printf("Priority of process %d changed to %d.\n", pid, priority);
    else
        perror("setpriority");
}

void task_manager()
{
    int choice;
    while (1)
    {
        printf("\nTask Manager\n");
        printf("1. List Processes\n");
        printf("2. Kill Process\n");
        printf("3. Change Process Priority\n");
        printf("4. Back to Main Menu\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice)
        {
        case 1:
            list_processes();
            break;
        case 2:
            kill_process();
            break;
        case 3:
            change_priority();
            break;
        case 4:
            return;
        default:
            printf("Invalid choice, try again.\n");
        }
    }
}
