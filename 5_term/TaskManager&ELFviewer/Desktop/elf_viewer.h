#ifndef ELF_VIEWER_H
#define ELF_VIEWER_H

void display_elf_headers(const char *filename);
void search_symbol(const char *filename, const char *symbol);
void elf_viewer();

#endif
