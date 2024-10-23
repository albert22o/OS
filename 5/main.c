

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"

int main() {
    // Создание пустого списка
    Student* studentList = NULL;

    // Пример данных для студентов
    int grades1[] = {5, 4, 5, 5, 3};
    int grades2[] = {3, 4, 2, 4, 5};

    // Создание и добавление студентов в список
    Student* student1 = createStudent("Оганесян Альберт Самвелович", grades1, "ул. Бориса-богаткова 63/1");
    addStudent(&studentList, student1);

    Student* student2 = createStudent("Лацук Андрей Юрьевич", grades2, "ул. Восход 9");
    addStudent(&studentList, student2);

    // Вывод информации о студентах
    printStudents(studentList);

    // Освобождение памяти
    freeStudents(studentList);

    return 0;
}
