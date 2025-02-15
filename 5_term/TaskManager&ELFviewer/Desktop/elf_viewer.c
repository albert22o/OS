#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <libelf.h>
#include <gelf.h>
#include <string.h>
#include "elf_viewer.h"

void display_elf_headers(const char *filename)
{
    int fd;
    Elf *elf;
    GElf_Ehdr ehdr;

    if (elf_version(EV_CURRENT) == EV_NONE)
    {
        fprintf(stderr, "ELF library initialization failed.\n");
        return;
    }

    fd = open(filename, O_RDONLY);
    if (fd < 0)
    {
        perror("open");
        return;
    }

    elf = elf_begin(fd, ELF_C_READ, NULL);
    if (!elf)
    {
        fprintf(stderr, "elf_begin() failed: %s\n", elf_errmsg(-1));
        close(fd);
        return;
    }

    if (gelf_getehdr(elf, &ehdr) == NULL)
    {
        fprintf(stderr, "Failed to get ELF header: %s\n", elf_errmsg(-1));
        elf_end(elf);
        close(fd);
        return;
    }

    printf("ELF Header:\n");
    printf("  Entry point: 0x%lx\n", (unsigned long)ehdr.e_entry);
    printf("  Machine: %d\n", ehdr.e_machine);
    printf("  Type: %d\n", ehdr.e_type);

    elf_end(elf);
    close(fd);
}

void search_symbol(const char *filename, const char *symbol)
{
    int fd;
    Elf *elf;
    Elf_Scn *scn = NULL;
    GElf_Shdr shdr;
    Elf_Data *data;

    if (elf_version(EV_CURRENT) == EV_NONE)
    {
        fprintf(stderr, "ELF library initialization failed.\n");
        return;
    }

    fd = open(filename, O_RDONLY);
    if (fd < 0)
    {
        perror("open");
        return;
    }

    elf = elf_begin(fd, ELF_C_READ, NULL);
    if (!elf)
    {
        fprintf(stderr, "elf_begin() failed: %s\n", elf_errmsg(-1));
        close(fd);
        return;
    }

    while ((scn = elf_nextscn(elf, scn)) != NULL)
    {
        if (gelf_getshdr(scn, &shdr) != &shdr)
            continue;
        if (shdr.sh_type == SHT_SYMTAB || shdr.sh_type == SHT_DYNSYM)
        {
            data = elf_getdata(scn, NULL);
            size_t count = shdr.sh_size / shdr.sh_entsize;
            for (size_t i = 0; i < count; ++i)
            {
                GElf_Sym sym;
                gelf_getsym(data, i, &sym);
                const char *name = elf_strptr(elf, shdr.sh_link, sym.st_name);
                if (name && strcmp(name, symbol) == 0)
                {
                    printf("Symbol found: %s at address 0x%lx\n", name, (unsigned long)sym.st_value);
                    elf_end(elf);
                    close(fd);
                    return;
                }
            }
        }
    }

    printf("Symbol not found.\n");
    elf_end(elf);
    close(fd);
}

void elf_viewer()
{
    char filename[256], symbol[256];
    int choice;

    printf("\nEnter ELF file path: ");
    scanf("%s", filename);

    while (1)
    {
        printf("\nELF Viewer\n");
        printf("1. Display ELF Headers\n");
        printf("2. Search Symbol\n");
        printf("3. Back to Main Menu\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice)
        {
        case 1:
            display_elf_headers(filename);
            break;
        case 2:
            printf("Enter symbol to search: ");
            scanf("%s", symbol);
            search_symbol(filename, symbol);
            break;
        case 3:
            return;
        default:
            printf("Invalid choice, try again.\n");
        }
    }
}
