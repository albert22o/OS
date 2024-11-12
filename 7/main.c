#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>

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
    void (*hello)();
    pid_t pid = getpid();

    // загрузка библиотеки
    handle = dlopen("./libmylib.so", RTLD_LAZY);
    if (!handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    // вызов функции из библиотеки
    hello = dlsym(handle, "hello");
    if (!hello)
    {
        fprintf(stderr, "%s\n", dlerror());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }
    hello();

    // печать карты памяти
    printf("\nMemory maps after loading the library:\n");
    print_memory_maps(pid);

    // выгрузка библиотеки
    dlclose(handle);

    // печать карты памяти
    printf("\nMemory maps after unloading the library:\n");
    print_memory_maps(pid);

    return 0;
}