#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"

// Функция для создания нового студента
Student *createStudent(const char *name, int grades[], const char *address)
{
    Student *newStudent = (Student *)malloc(sizeof(Student));
    if (newStudent == NULL)
    {
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
void addStudent(Student **head, Student *newStudent)
{
    newStudent->next = *head;
    *head = newStudent;
}

// Функция для вывода информации о студентах
void printStudents(const Student *head)
{
    const Student *current = head;
    while (current != NULL)
    {
        printf("ФИО: %s\n", current->name);
        printf("Оценки: ");
        for (int i = 0; i < 5; i++)
        {
            printf("%d ", current->grades[i]);
        }
        printf("\n");
        printf("Адрес: %s\n", current->address);
        printf("-------------------------\n");
        current = current->next;
    }
}

// Функция для освобождения памяти списка
void freeStudents(Student *head)
{
    Student *current = head;
    while (current != NULL)
    {
        Student *temp = current;
        current = current->next;
        free(temp);
    }
}
