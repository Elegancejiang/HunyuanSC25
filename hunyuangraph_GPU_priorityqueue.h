#ifndef _H_GPU_PRIORITYQUEUE
#define _H_GPU_PRIORITYQUEUE

#include "hunyuangraph_struct.h"


// __device__ int get_random_number_range(int range, curandState *localState) 
// {
//     float randNum = curand_uniform(localState);
//     return (int)(randNum * range);
// }

// typedef struct 
// {
// 	int nnodes;
// 	int maxnodes;
// 	int *key;
// 	int *val;
// 	int *locator;
// } priority_queue_t;

// __device__ void priority_queue_Init(priority_queue_t *queue, int maxnodes)
// {
// 	queue->nownodes = 0;
// 	queue->maxnodes = maxnodes;
// 	// queue->heap     = (node_t *)check_malloc(sizeof(node_t) * maxnodes, "priority_queue_Init: heap");
// 	// queue->locator  = (int *)check_malloc(sizeof(int) * maxnodes, "priority_queue_Init: locator");
// 	for(int i = 0;i < maxnodes;i++)
// 		queue->locator[i] = -1;
// }

// __device__ priority_queue_t *priority_queue_Create(int maxnodes)
// {
// 	// priority_queue_t *queue; 

// 	// queue = (priority_queue_t *)check_malloc(sizeof(priority_queue_t), "priority_queue_Create: queue");
// 	priority_queue_Init(queue, maxnodes);

// 	return queue;
// }

#endif