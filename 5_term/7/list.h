#ifndef LIST_H
#define LIST_H

// Определение структуры для студента
typedef struct Student
{
    char name[100];       // ФИО студента
    int grades[5];        // Оценки (допустим, 5 оценок)
    char address[200];    // Адрес проживания
    struct Student *next; // Указатель на следующий элемент списка
} Student;

// Функция для создания нового студента
Student *createStudent(const char *name, int grades[], const char *address);
// Функция для добавления студента в начало списка
void addStudent(Student **head, Student *newStudent);

// Функция для вывода информации о студентах
void printStudents(const Student *head);

// Функция для освобождения памяти списка
void freeStudents(Student *head);

#endif