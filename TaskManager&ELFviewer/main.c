#include <stdio.h>
#include <stdlib.h>
#include "task_manager.h"
#include "elf_viewer.h"

int main()
{
    int choice;
    while (1)
    {
        printf("\nTask Manager and ELF Viewer\n");
        printf("1. Task Manager\n");
        printf("2. ELF Viewer\n");
        printf("3. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice)
        {
        case 1:
            task_manager();
            break;
        case 2:
            elf_viewer();
            break;
        case 3:
            printf("Exiting...\n");
            return 0;
        default:
            printf("Invalid choice, try again.\n");
        }
    }
    return 0;
}
