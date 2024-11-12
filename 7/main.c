#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"
#include <dlfcn.h>
#include <unistd.h>

// Вывод карт памяти по pid
void print_memory_maps(pid_t pid)
{
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE *file = fopen(path, "r");

    if (!file)
        perror("Error: couldn't open the file!");

    char line[256];

    while (fgets(line, sizeof(line), file))
        printf("%s", line);

    fclose(file);
}

int main()
{
    void *handle;
    Student *(*createStudent)();
    void (*addStudent)();
    void (*printStudents)();
    void (*freeStudents)();
    pid_t pid = getpid();

    // загрузка библиотеки
    handle = dlopen("./liblist.so", RTLD_LAZY);
    if (!handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    // Вызов функций из библиотеки
    createStudent = dlsym(handle, "createStudent");
    if (!createStudent)
    {
        fprintf(stderr, "%s\n", dlerror());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }

    addStudent = dlsym(handle, "addStudent");
    if (!addStudent)
    {
        fprintf(stderr, "%s\n", dlerror());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }

    printStudents = dlsym(handle, "printStudents");
    if (!printStudents)
    {
        fprintf(stderr, "%s\n", dlerror());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }

    freeStudents = dlsym(handle, "freeStudents");
    if (!freeStudents)
    {
        fprintf(stderr, "%s\n", dlerror());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }

    // ------------ Выполнение кода программы ------------
    // Создание пустого списка
    Student *studentList = NULL;

    // Пример данных для студентов
    int grades1[] = {5, 4, 5, 5, 3};
    int grades2[] = {3, 4, 2, 4, 5};

    // Создание и добавление студентов в список
    Student *student1 = createStudent("Оганесян Альберт Самвелович", grades1, "ул. Бориса-богаткова 63/1");
    addStudent(&studentList, student1);

    Student *student2 = createStudent("Лацук Андрей Юрьевич", grades2, "ул. Восход 9");
    addStudent(&studentList, student2);

    // Вывод информации о студентах
    printStudents(studentList);

    // Освобождение памяти
    freeStudents(studentList);

    // ------------ Контроль карт памяти ------------
    // печать карты памяти после загрузки
    printf("\nКарты памяти после загрузки библиотеки:\n");
    print_memory_maps(pid);

    // выгрузка библиотеки
    dlclose(handle);

    // печать карты памяти после выгрузки
    printf("\nКарты памяти после выгрузки библиотеки:\n");
    print_memory_maps(pid);

    return 0;
}
