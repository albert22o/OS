#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(int argc, char** argv){
  const char* elfFile=argv[1];
  Elf64_Ehdr header;
  Elf64_Shdr sheader;
  Elf64_Shdr symtab;
  Elf64_Shdr strtab;
  Elf64_Shdr shstrtab;
  Elf64_Sym sym;
  char sname[32];
  char sectionName[32];
  int i;
  FILE* file = fopen(elfFile, "rb");
  if(file==NULL){
    fprintf(stderr, "Error opening file %s\n", elfFile);
    return 1;
  }
  fread(&header, sizeof(header), 1, file);

   fseek(file, header.e_shoff, SEEK_SET);

  fread(&sheader, sizeof(sheader), 1, file);
  for(i=0; i<header.e_shnum;i++){

    fseek(file,header.e_shoff+header.e_shentsize*i, SEEK_SET);

    fread(&sheader, sizeof(sheader), 1, file);
    if(i==4)
      symtab=(Elf64_Shdr)sheader;
    if(i==5)
     strtab=(Elf64_Shdr)sheader;
  }

  fseek(file, header.e_shoff + header.e_shentsize * header.e_shstrndx, SEEK_SET);
  fread(&shstrtab, sizeof(shstrtab), 1, file);

  printf("Section Names:\n");
  for (i = 0; i < header.e_shnum; i++) {
      fseek(file, header.e_shoff + header.e_shentsize * i, SEEK_SET);
      fread(&sheader, sizeof(sheader), 1, file);

      fseek(file, shstrtab.sh_offset + sheader.sh_name, SEEK_SET);
      fread(sectionName, 1, sizeof(sectionName) - 1, file);
      sectionName[31] = '\0';

      printf("Section %d: %s\n", i, sectionName);
  }

  for (i = 0; i < header.e_shnum; i++) {
    fseek(file, header.e_shoff + header.e_shentsize * i, SEEK_SET);
    if (fread(&sheader, sizeof(sheader), 1, file) != 1) {
      fprintf(stderr, "Failed to read section header %d\n", i);
      fclose(file);
      return 1;
    }

    // Get section name
    fseek(file, shstrtab.sh_offset + sheader.sh_name, SEEK_SET);
    fread(sectionName, 1, sizeof(sectionName) - 1, file);
    sectionName[31] = '\0';

    // Check for .symtab and .strtab
    if (strcmp(sectionName, ".symtab") == 0) {
      symtab = sheader;
    } else if (strcmp(sectionName, ".strtab") == 0) {
      strtab = sheader;
    }
  }
  
  for(i=0;i<symtab.sh_size / symtab.sh_entsize;i++)
    {
     fseek(file,symtab.sh_offset + symtab.sh_entsize*i, SEEK_SET);    
     fread(&sym, sizeof(Elf64_Sym), 1, file);
     fseek(file,strtab.sh_offset+sym.st_name, SEEK_SET);         
     fread(sname, 1,32, file);
     fprintf(stdout, "%d\t%lld\t%u\t%u\t%hd\t%s\n", i,
              sym.st_size, 
              ELF64_ST_TYPE(sym.st_info),
              ELF64_ST_BIND(sym.st_info), 
              sym.st_shndx, sname);     
   }
  

  return 0;
}

