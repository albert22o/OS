## Содержание документа:
- #### [1. Дисчпетчер задач](#пример-использования-диспетчера-задач)
- #### [2. ELF Viewer](#пример-использования-elf-viewer)

### Пример использования диспетчера задач
#### 1. Запуск программы
Компиляция программы:
```
gcc main.c task_manager.c elf_viewer.c -o task_elf_manager -lelf
```
Использование sudo: Для завершения процессов и изменения их приоритета программа требует прав суперпользователя. Запуск:
```
sudo ./task_elf_manager
```
При запуске появится главное меню:
```
Task Manager and ELF Viewer
1. Task Manager
2. ELF Viewer
3. Exit
Enter your choice: 1
```
#### 2. Работа с Task Manager
После выбора "1" вы перейдёте в меню диспетчера задач:
```
Task Manager
1. List Processes
2. Kill Process
3. Change Process Priority
4. Back to Main Menu
Enter your choice: 
```
#### 3. Список процессов
Выберите **1. List Processes**, чтобы увидеть все текущие процессы:
```
PID	PPID	Memory	State	Priority	Name
-----------------------------------------------------------
1	0	12345	S	0	(init)
42	1	23456	S	-10	(systemd)
2345	42	78901	R	0	(ssh)
3456	2345	45678	S	5	(bash)
...
```
**Описание столбцов:**
- **PID:** Идентификатор процесса.
- **PPID:** Идентификатор родительского процесса.
- **Memory:** Используемая память.
- **State:** Состояние процесса:
    - **R** — выполняется.
    - **S** — спит.
    - **Z** — зомби.
- **Priority:** Приоритет процесса (диапазон от -20 до 19).
- **Name:** Имя процесса.
#### 4. Завершение процесса
Чтобы завершить процесс, выберите 2. **Kill Process:**
```
Enter PID to kill: 3456
```
Если завершение успешно:
```
Process 3456 killed.
```
Если процесс не удалось завершить:
```
kill: Operation not permitted
```
**Убедитесь, что вы запускаете программу с правами *sudo*, чтобы завершить системные процессы.**
#### 5. Изменение приоритета процесса
Выберите **3. Change Process Priority**:
```
Enter PID to change priority: 2345
Enter new priority (-20 to 19): -5
```
Если изменение успешно:
```
Priority of process 2345 changed to -5.
```
Если возникла ошибка:
```
setpriority: Operation not permitted
```
**Для изменения приоритета системных процессов также требуется *sudo*.**
#### 6. Возврат в главное меню
Выберите **4. Back to Main Menu**, чтобы вернуться в основное меню программы:
```
Task Manager
1. List Processes
2. Kill Process
3. Change Process Priority
4. Back to Main Menu
Enter your choice: 4
```
---
### Пример использования ELF Viewer
#### 1. Подготовка ELF-файла для теста
Создадим простую программу на C, скомпилируем её и используем для анализа:
**Код** `test.c`:
```
#include <stdio.h>

void test_function() {
    printf("Hello from test_function!\n");
}

int main() {
    printf("Hello, World!\n");
    test_function();
    return 0;
}
```
**Компиляция программы:**
```
gcc -o test test.c
```
После этого у нас будет ELF-файл test.
#### 2. Запуск ELF Viewer
Запустим наше приложение и выберем ELF Viewer.
```
./task_elf_manager
```
**Меню:**
```
Task Manager and ELF Viewer
1. Task Manager
2. ELF Viewer
3. Exit
Enter your choice: 2
```
#### 3. Анализ ELF-файла
**Введите путь к ELF-файлу:**
```
Enter ELF file path: ./test
```
**Меню ELF Viewer:**
```
ELF Viewer
1. Display ELF Headers
2. Search Symbol
3. Back to Main Menu
Enter your choice: 1
```
**Вывод заголовков ELF-файла:**
```
ELF Header:
  Entry point: 0x401000
  Machine: 62
  Type: 2
```
- **Entry point**: Указывает на начальный адрес выполнения программы (в данном случае 0x401000).
- **Machine**: Тип архитектуры (62 соответствует x86-64).
- **Type**: Тип ELF-файла (2 — выполняемый файл).
#### 4. Поиск символов
**Поиск функции test_function:**
```
ELF Viewer
1. Display ELF Headers
2. Search Symbol
3. Back to Main Menu
Enter your choice: 2

Enter symbol to search: test_function
```
**Результат:**
```
Symbol found: test_function at address 0x401126
```
Эта функция найдена в таблице символов, и её адрес — 0x401126.
#### 5. Возврат в меню
**Чтобы вернуться в главное меню, выберите:**
```
3. Back to Main Menu
```