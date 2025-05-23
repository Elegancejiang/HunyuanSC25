#ifndef MEMOTY_H
#define MEMOTY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <stdbool.h>
#include <sys/time.h>
#include "common.h"
#include "define.h"
#include "struct.h"
#include "timer.h"

memory_manage *memorymanage = NULL;
char *name = NULL;
Hunyuan_real_t log_time;
struct timeval start_log;
struct timeval end_log;

bool find_between_last_slash_and_dotgraph(const char *filename) 
{
    name = (char *)malloc(sizeof(char) * 128);
    const char *last_slash_pos = strrchr(filename, '/');
    const char *dotgraph_pos = strstr(filename, ".graph");
    
    // 确保找到了 '/' 和 '.graph'，并且 '/' 在 '.graph' 之前
    if (last_slash_pos != NULL && dotgraph_pos != NULL && last_slash_pos < dotgraph_pos) 
    {
        // 计算开始复制字符的位置
        const char *start_pos = last_slash_pos + 1; // 跳过 '/'
        
        // 计算结束复制的位置
        const char *end_pos = dotgraph_pos; // '.graph' 前的位置
        
        // 计算需要复制的字符数量
        size_t len = end_pos - start_pos;

        strncpy(name, start_pos, len);
        name[len] = '\0'; // 确保字符串以 null 结尾

        strcat(name, "_log.txt");
        printf("name=%s\n",name);

        gettimeofday(&start_log, NULL);

        return true;
    }

    return false;
}

/*************************************************************************/
/* return:
        *-1 -> already init
        * 0 -> Memory init failed
        * 1 -> init successfully
*/
/**************************************************************************/
Hunyuan_int_t init_memery_manage(char *filename)
{
    //  already init
    if(memorymanage != NULL)
        return -1;

    memorymanage = (memory_manage *)malloc(sizeof(memory_manage));
    if(memorymanage == NULL)
    {
        printf("***Memory allocation failed for memorymanage.");

		exit(1);

		return 0;
    }

    memorymanage->all_block = 1024;
    memorymanage->used_block = 0;
    memorymanage->now_memory = 0;
    memorymanage->max_memory = 0;
    memorymanage->memoryblock = (memory_block *)malloc(sizeof(memory_block) * memorymanage->all_block);
    if(memorymanage->memoryblock == NULL)
    {
        free(memorymanage);
        memorymanage = NULL;
        char *error_message = (char *)malloc(sizeof(char) * 128);
		sprintf(error_message, "***Memory allocation failed for memoryblock.");
		error_exit(error_message);
    }
    for(Hunyuan_int_t i = 0;i < memorymanage->all_block;i++)
    {
        memorymanage->memoryblock[i].ptr = NULL;
        memorymanage->memoryblock[i].nbytes = 0;
    }

    // if(filename != NULL)
    //     find_between_last_slash_and_dotgraph(filename);

    return 1;
}

// void log(int task_type, size_t nbytes, void *ptr)
// {
//     struct timeval now;
//     gettimeofday(&now, NULL);

//     fprintf("log.txt", "time=%ld task_type=%d nbytes=%d ptr=%p\n",now.tv_sec*1000000 + now.tv_usec,task_type,nbytes,ptr);
// }

void log_memory(Hunyuan_int_t task_type, size_t nbytes, void *ptr, char *message) 
{
    // 打开文件
    FILE *file = fopen(name, "a"); // "a" 模式表示追加到文件末尾
    if (file == NULL) 
    {
        perror("Failed to open log file");
        return;
    }

    gettimeofday(&end_log,NULL);
    log_time = (end_log.tv_sec - start_log.tv_sec) * 1000 + (end_log.tv_usec - start_log.tv_usec) / 1000.0;

    // 写入日志
    // fprintf(file, "Memory Log --------------------------------------------------------\n");
    fprintf(file, "%"PRREAL" %"PRIDX" %10zu\n", log_time, task_type, nbytes);
    // fprintf(file, "time=%"PRREAL" task_type=%"PRIDX" nbytes=%10zu\n", log_time, task_type, nbytes);
    // fprintf(file, "time=%"PRREAL" task_type=%"PRIDX" ptr=%p nbytes=%10zu located at %s\n", log_time, task_type, ptr, nbytes, message);
    // fprintf(file, "-------------------------------------------------------------------\n");

    // 关闭文件
    fclose(file);
}

void add_memory_block(void *ptr, size_t nbytes, char *message)
{
    // need to realloc
    if(memorymanage->used_block >= memorymanage->all_block)
    {
        // printf("double\n");
        // double
        memorymanage->all_block *= 2;
        memorymanage->memoryblock = (memory_block *)realloc(memorymanage->memoryblock,sizeof(memory_block) * memorymanage->all_block);
        if(memorymanage->memoryblock == NULL)
        {
            char *error_message = (char *)malloc(sizeof(char) * 128);
			sprintf(error_message, "***Memory allocation failed for memoryblock.");
			error_exit(error_message);
        }
        for(Hunyuan_int_t i = memorymanage->all_block / 2;i < memorymanage->all_block;i++)
        {
            memorymanage->memoryblock[i].ptr = NULL;
            memorymanage->memoryblock[i].nbytes = 0;
        }
    }

    Hunyuan_int_t choose = memorymanage->used_block;
    memorymanage->memoryblock[choose].ptr = ptr;
    memorymanage->memoryblock[choose].nbytes = nbytes;
    memorymanage->now_memory += nbytes;
    memorymanage->max_memory = lyj_max(memorymanage->max_memory, memorymanage->now_memory);
    memorymanage->used_block ++;

    // log_memory(1, memorymanage->now_memory, ptr, message);

    // printf("add check_malloc for %s ptr=%p nbytes=%zu\n",message,ptr,nbytes);

    return ;
}

void update_memory_block(void *ptr, void *oldptr, size_t nbytes, size_t old_nbytes, char *message)
{
    for(Hunyuan_int_t i = 0;i < memorymanage->all_block;i++)
    {
        if(memorymanage->memoryblock[i].ptr == oldptr)
        {
            // size_t old_nbytes = memorymanage->memoryblock[i].nbytes;
            memorymanage->now_memory -=  memorymanage->memoryblock[i].nbytes;
            memorymanage->memoryblock[i].nbytes = nbytes;
            memorymanage->now_memory += nbytes;
            memorymanage->max_memory = lyj_max(memorymanage->max_memory, memorymanage->now_memory);
            memorymanage->memoryblock[i].ptr = ptr;

            // log_memory(2, memorymanage->now_memory, ptr, message);

            // printf("update check_realloc for %s ptr=%p oldptr=%p nbytes=%zu\n",message,ptr,oldptr,nbytes);

            return ;
        }
    }

    char *error_message = (char *)malloc(sizeof(char) * 128);
	sprintf(error_message, "***update_memory_block failed for memoryblock.");
	error_exit(error_message);
}

void delete_memory_block(void *ptr, char *message)
{
    Hunyuan_int_t choose = memorymanage->used_block - 1;
    for(Hunyuan_int_t i = choose;i >= 0;i--)
    {
        if(memorymanage->memoryblock[i].ptr == ptr)
        {
            memorymanage->now_memory -= memorymanage->memoryblock[i].nbytes;
            // printf("delete check_free for %s ptr=%p nbytes=%zu\n",message,ptr,memorymanage->memoryblock[i].nbytes);
            memorymanage->max_memory = lyj_max(memorymanage->max_memory, memorymanage->now_memory);
            memorymanage->used_block--;
            if(i != choose)
            {
                memorymanage->memoryblock[i].ptr = memorymanage->memoryblock[choose].ptr;
                memorymanage->memoryblock[i].nbytes = memorymanage->memoryblock[choose].nbytes;
                memorymanage->memoryblock[choose].ptr = NULL;
                memorymanage->memoryblock[choose].nbytes = 0;
            }

            else 
            {
                memorymanage->memoryblock[i].ptr = NULL;
                memorymanage->memoryblock[i].nbytes = 0;
            }

            // log_memory(3, memorymanage->now_memory, ptr, message);

            return ;
        }
        // printf("loop check_free for %s ptr=%p\n",message,ptr);
        // for(Hunyuan_int_t i = 0;i <= choose;i++)
        // {
        //     printf("i=%"PRIDX" ptr=%p nbytes=%zu\n",i,memorymanage->memoryblock[i].ptr, memorymanage->memoryblock[i].nbytes);
        // }
    }
    
    char *error_message = (char *)malloc(sizeof(char) * 128);
	sprintf(error_message, "***delete_memory_block failed for memoryblock.");
	error_exit(error_message);
}

void free_memory_block()
{
    if(memorymanage->memoryblock != NULL)
    {
        free(memorymanage->memoryblock);
        memorymanage->memoryblock = NULL;
    }
    if(memorymanage->memoryblock != NULL)
    {
        char *error_message = (char *)malloc(sizeof(char) * 128);
		sprintf(error_message, "***Memory free failed for memoryblock.");
		error_exit(error_message);
    }

    if(memorymanage != NULL)
    {
        // printf("memorymanage->now_memory=%zu\n",memorymanage->now_memory);
        free(memorymanage);
        memorymanage = NULL;
    }
    if(memorymanage != NULL)
    {
        char *error_message = (char *)malloc(sizeof(char) * 128);
		sprintf(error_message, "***Memory free failed for memoryblock.");
		error_exit(error_message);
    }
}

/*************************************************************************/
/* function:
        *Check whether memory is allocated successfully
        *Compute the amount of memory used
        *The request space poHunyuan_int_ter type is set to void
*/
/**************************************************************************/
void *check_malloc(size_t nbytes, char *message)
{
	void *ptr = NULL;
    
    // need?
	if (nbytes == 0)
		nbytes += 4;  /* Force mallocs to actually allocate some memory */

	ptr = (void *)malloc(nbytes);
    
	if (ptr == NULL) 
	{
		// printf("   Current memory used:  %10zu bytes\n",memorymanage->now_memory);
		// printf("   Maximum memory used:  %10zu bytes\n",memorymanage->max_memory);
		// printf("***Memory allocation failed for %s. Requested size: %zu bytes", message, nbytes);
        fprintf(stderr, "Current memory used: %zu bytes\n", memorymanage->now_memory);
        fprintf(stderr, "Maximum memory used: %zu bytes\n", memorymanage->max_memory);
        fprintf(stderr, "***Memory allocation failed for %s. Requested size: %zu bytes\n", message, nbytes);
		return NULL;
	}

    // printf("ptr=%p %s\n",ptr,message);

    CONTROL_COMMAND(control, ALL_Time, gettimebegin(&start_malloc, &end_malloc, &time_malloc));
    // add_memory_block(ptr,nbytes,message);
    memorymanage->now_memory += nbytes;
    // printf("ptr=%p malloc=%s nbytes=%"PRIDX"\n",ptr, message, nbytes);
    // log_memory(1, memorymanage->now_memory, ptr, message);
    CONTROL_COMMAND(control, ALL_Time, gettimeend(&start_malloc, &end_malloc, &time_malloc));
	return ptr;
}

/*************************************************************************/
/* function:
        *Check whether memory is allocated successfully
        *Compute the amount of memory used
        *The request space poHunyuan_int_ter type is set to void
*/
/**************************************************************************/
void *check_realloc(void *oldptr, size_t nbytes, size_t old_nbytes, char *message)
{
	void *ptr = NULL;

	// need?
	if (nbytes == 0)
		nbytes += 4;  /* Force mallocs to actually allocate some memory */

  	ptr = (void *)realloc(oldptr, nbytes);

	if (ptr == NULL) 
	{
        printf("   Current memory used:  %10zu Bytes\n",memorymanage->now_memory);
		printf("   Maximum memory used:  %10zu Bytes\n",memorymanage->max_memory);
		printf("***Memory allocation failed for %s. Requested size: %zu bytes", message, nbytes);

		exit(1);

		return NULL;
	}

    // update_memory_block(ptr,oldptr,nbytes,old_nbytes,message);
    memorymanage->now_memory -= old_nbytes;
    memorymanage->now_memory += nbytes;
    // printf("ptr=%p realloc=%s nbytes=%"PRIDX"\n",ptr, message, nbytes);
    // log_memory(2, memorymanage->now_memory, ptr, message);

	return ptr;
}

/*************************************************************************/
/* function:
        *Check whether memory is freed successfully
        *Compute the amount of memory used
*/
/**************************************************************************/
void check_free(void *ptr, size_t nbytes, char *message)
{
	if (ptr != NULL) 
	{
        if (nbytes == 0)
		    nbytes += 4;
        // printf("check_free for %s\n",message);
		free(ptr);
        CONTROL_COMMAND(control, ALL_Time, gettimebegin(&start_free, &end_free, &time_free));
		// delete_memory_block(ptr,message);
        memorymanage->now_memory -= nbytes;
        // printf("ptr=%p free=%s nbytes=%"PRIDX"\n",ptr, message, nbytes);
        // log_memory(3, memorymanage->now_memory, ptr, message);
        CONTROL_COMMAND(control, ALL_Time, gettimeend(&start_free, &end_free, &time_free));
        ptr = NULL;
	}
}

void PrintMemory()
{
    printf("\nMemory Management --------------------------------------------------\n");
	printf("      Malloc:              %10.3"PRREAL" ms\n", time_malloc);
	printf("      free:                %10.3"PRREAL" ms\n", time_free);
	printf("      Current memory used:  %10zu Bytes\n",memorymanage->now_memory);
	printf("      Maximum memory used:  %10zu Bytes\n",memorymanage->max_memory);
    printf("      Current memory block used:  %10"PRIDX" block\n",memorymanage->used_block);
    printf("      Maximum memory block used:  %10"PRIDX" block\n",memorymanage->all_block);
    printf("-------------------------------------------------------------------\n");
}

void exam_memory()
{
    for(Hunyuan_int_t i = 0;i < memorymanage->all_block;i++)
    {
        if(memorymanage->memoryblock[i].ptr != NULL)
            printf("memorymanage->memoryblock[i].ptr=%p nbyte=%zu\n",memorymanage->memoryblock[i].ptr,memorymanage->memoryblock[i].nbytes);
    }
}



#endif