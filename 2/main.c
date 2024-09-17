#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Определение структуры для студента
typedef struct Student {
    char name[100];        // ФИО студента
    int grades[5];         // Оценки (допустим, 5 оценок)
    char address[200];     // Адрес проживания
    struct Student* next;  // Указатель на следующий элемент списка
} Student;

// Функция для создания нового студента
Student* createStudent(const char* name, int grades[], const char* address) {
    Student* newStudent = (Student*)malloc(sizeof(Student));
    if (newStudent == NULL) {
        printf("Ошибка выделения памяти\n");
        exit(1);
    }
    strncpy(newStudent->name, name, sizeof(newStudent->name));
    memcpy(newStudent->grades, grades, sizeof(newStudent->grades));
    strncpy(newStudent->address, address, sizeof(newStudent->address));
    newStudent->next = NULL;
    return newStudent;
}

// Функция для добавления студента в начало списка
void addStudent(Student** head, Student* newStudent) {
    newStudent->next = *head;
    *head = newStudent;
}

// Функция для вывода информации о студентах
void printStudents(const Student* head) {
    const Student* current = head;
    while (current != NULL) {
        printf("ФИО: %s\n", current->name);
        printf("Оценки: ");
        for (int i = 0; i < 5; i++) {
            printf("%d ", current->grades[i]);
        }
        printf("\n");
        printf("Адрес: %s\n", current->address);
        printf("-------------------------\n");
        current = current->next;
    }
}

// Функция для освобождения памяти списка
void freeStudents(Student* head) {
    Student* current = head;
    while (current != NULL) {
        Student* temp = current;
        current = current->next;
        free(temp);
    }
}

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
