#include <stdio.h>

void test_function()
{
    printf("Hello from test_function!\n");
}

int main()
{
    while (1)
    {
        printf("Hello, World!\n");
        test_function();
    }
    return 0;
}
