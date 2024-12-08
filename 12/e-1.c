#include <semaphore.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
int main( void ) {
int n=0;
int fd;
char* sh;
sem_t *sem;
fd=shm_open("/common_region",
O_RDWR | O_CREAT,
S_IRUSR | S_IWUSR | S_IRGRP);
if (fd == -1)
fprintf(stderr, "shm_open\n");
ftruncate(fd, 6);
sh=(char*)mmap(0,6,
PROT_READ | PROT_WRITE,
MAP_SHARED, fd, 0);
if (sh == MAP_FAILED)
fprintf(stderr, "mmap\n");
memset(sh,0,6);
sem=sem_open("/common_sem", O_CREAT,
S_IRUSR | S_IWUSR | S_IRGRP, 1);
if (sem == SEM_FAILED)
fprintf(stderr, "sem_open");
while(n++<200){
sem_wait(sem);
//write(fileno(stdout),sh, 6);
printf("String: %s\n",sh);
sem_post(sem);
usleep(100);
}
shm_unlink("/common_region");
munmap(sh, 6);
sem_unlink("/common_sem");
sem_close(sem);
return 0;
}