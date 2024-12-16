#ifndef _H_GPU_INITIALPARTITION
#define _H_GPU_INITIALPARTITION

#include "hunyuangraph_struct.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_admin.h"
#include "hunyuangraph_GPU_priorityqueue.h"

#include <cuda_runtime.h>
#include <cstdint> 
// #include "device_launch_parameters.h"
#include "curand_kernel.h"

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err));
        // You may also choose to exit or handle the error differently here.
    }
}

// typedef struct 
// {
// 	int nownodes;
// 	int maxnodes;
// 	int *key;
// 	int *val;
// 	int *locator;
// } priority_queue_t;

__device__ void priority_queue_Init(priority_queue_t *queue, int maxnodes)
{
	queue->nownodes = 0;
	queue->maxnodes = maxnodes;
	for(int i = 0;i < maxnodes;i++)
		queue->locator[i] = -1;
}

__device__ void priority_queue_Reset(priority_queue_t *queue, int maxnodes)
{
	for (int i = queue->nownodes - 1; i >= 0; i--)
		queue->locator[queue->val[i]] = -1;
	queue->nownodes = 0;
    queue->maxnodes = maxnodes;
}

__device__ int priority_queue_Length(priority_queue_t *queue)
{
	return queue->nownodes;
}

__device__ int priority_queue_Insert(priority_queue_t *queue, int node, int key)
{
	int i, j;

	i = queue->nownodes++;
	while (i > 0) 
	{
		j = (i - 1) >> 1;
		if (key > queue->key[j]) 
		{
            queue->val[i] = queue->val[j];
            queue->key[i] = queue->key[j];
			queue->locator[queue->val[i]] = i;
			i = j;
		}
		else
			break;
	}
  
	queue->key[i]   = key;
	queue->val[i]   = node;
	queue->locator[node] = i;

	return 0;
}

__device__ int priority_queue_Delete(priority_queue_t *queue, int node)
{
	int i, j, nownodes;
	int newkey, oldkey;

	i = queue->locator[node];
	queue->locator[node] = -1;

	if (--queue->nownodes > 0 && queue->val[queue->nownodes] != node) 
	{
		node   = queue->val[queue->nownodes];
		newkey = queue->key[queue->nownodes];
		oldkey = queue->key[i];

		if (newkey > oldkey) 
		{ /* Filter-up */
			while (i > 0) 
			{
				j = (i - 1) >> 1;
				if (newkey > queue->key[j]) 
				{
                    queue->val[i] = queue->val[j];
                    queue->key[i] = queue->key[j];
					queue->locator[queue->val[i]] = i;
					i = j;
				}
				else
					break;
			}
		}
		else 
		{ /* Filter down */
			nownodes = queue->nownodes;
			while ((j = (i << 1) + 1) < nownodes) 
			{
				if (queue->key[j] > newkey) 
				{
					if (j + 1 < nownodes && queue->key[j + 1] > queue->key[j])
						j++;
                    queue->val[i] = queue->val[j];
                    queue->key[i] = queue->key[j];
					queue->locator[queue->val[i]] = i;
					i = j;
				}
				else if (j + 1 < nownodes && queue->key[j + 1] > newkey)
				{
					j++;
                    queue->val[i] = queue->val[j];
                    queue->key[i] = queue->key[j];
					queue->locator[queue->val[i]] = i;
					i = j;
				}
				else
					break;
			}
		}

		queue->key[i] = newkey;
		queue->val[i] = node;
		queue->locator[node] = i;
	}

	return 0;
}

__device__ void priority_queue_Update(priority_queue_t *queue, int node, int newkey)
{
	int i, j, nownodes;
	int oldkey;

	oldkey = queue->key[queue->locator[node]];

	i = queue->locator[node];

	if (newkey > oldkey) 
	{ /* Filter-up */
		while (i > 0) 
		{
			j = (i - 1) >> 1;
			if (newkey > queue->key[j]) 
			{
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else
				break;
		}
	}
	else 
	{ /* Filter down */
		nownodes = queue->nownodes;
		while ((j = (i << 1) + 1) < nownodes) 
		{
			if (queue->key[j] > newkey)
			{
				if (j + 1 < nownodes && queue->key[j + 1] > queue->key[j])
					j++;
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else if (j + 1 < nownodes && queue->key[j + 1] > newkey) 
			{
				j++;
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else
				break;
		}
	}

	queue->key[i] = newkey;
	queue->val[i] = node;
	queue->locator[node] = i;

	return;
}

__device__ int priority_queue_GetTop(priority_queue_t *queue)
{
	int i, j;
	int vtx, node;
	int key;

	if (queue->nownodes == 0)
		return -1;

	queue->nownodes--;

	vtx = queue->val[0];
	queue->locator[vtx] = -1;

	if ((i = queue->nownodes) > 0) 
	{
		key  = queue->key[i];
		node = queue->val[i];
		i = 0;
		while ((j = 2 * i + 1) < queue->nownodes) 
		{
			if (queue->key[j] > key) 
			{
				if (j + 1 < queue->nownodes && queue->key[j + 1] > queue->key[j])
					j = j + 1;
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else if (j + 1 < queue->nownodes && queue->key[j + 1] > key) 
			{
				j = j + 1;
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else
				break;
		}

		queue->key[i] = key;
		queue->val[i] = node;
		queue->locator[node] = i;
	}

	return vtx;
}

__global__ void initializeCurand(unsigned long long seed, unsigned long long offset, unsigned long long nvtxs, curandStateXORWOW *devStates) 
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, offset, nvtxs, &devStates[ii]);
}

__device__ int get_random_number_range(int range, curandState *localState) 
{
    float randNum = curand_uniform(localState);
    return (int)(randNum * range);
}

__device__ void compute_v_ed_id_bnd(int v, int *xadj, int *adjncy, int *adjwgt, hunyuangraph_int8_t *where, int *ed, int *id, hunyuangraph_int8_t *bnd)
{
    int begin, end, ted, tid;
    hunyuangraph_int8_t me, other;
    begin = xadj[v];
    end = xadj[v + 1];
    me = where[v];
    ted = 0;
    tid = 0;

    if(begin == end)
    {
        bnd[v] = 1;
        ed[v] = 0;
        id[v] = 0;
        return ;
    }

    for(int i = begin;i < end; i++)
    {
        int wgt = adjwgt[i];
        int j = adjncy[i];
        other = where[j];

        if(me != other) 
            ted += wgt;
        else 
            tid += wgt;
    }

    ed[v] = ted;
    id[v] = tid;

    if(ted > 0)
        bnd[v] = 1;
    else
        bnd[v] = 0;
}

__device__ void warpReduction(volatile int *reduce_num, int tid ,int blocksize)
{
    if(blocksize >= 64) reduce_num[tid] += reduce_num[tid + 32];
    if(blocksize >= 32) reduce_num[tid] += reduce_num[tid + 16];
    if(blocksize >= 16) reduce_num[tid] += reduce_num[tid + 8];
    if(blocksize >= 8) reduce_num[tid] += reduce_num[tid + 4];
    if(blocksize >= 4) reduce_num[tid] += reduce_num[tid + 2];
    if(blocksize >= 2) reduce_num[tid] += reduce_num[tid + 1];
}

__device__ int hunyuangraph_gpu_int_min(int first, int second)
{
    if(first <= second)
        return first;
    else
        return second;
}

__device__ int hunyuangraph_gpu_int_max(int first, int second)
{
    if(first >= second)
        return first;
    else
        return second;
}

__device__ int hunyuangraph_gpu_int_abs(int a)
{
    if(a < 0)
        return -a;
    else
        return a;
}

__device__ void hunyuangraph_gpu_int_swap(int *a, int *b)
{
    int t = a[0];
    a[0] = b[0];
    b[0] = t;
}

__device__ void FM_2way_cut_refinement(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, hunyuangraph_int8_t *where, int *pwgts, int *ed, int *id, \
    hunyuangraph_int8_t *bnd, hunyuangraph_int8_t *moved, int *swaps, int tvwgt, int edgecut, double ntpwgts0, priority_queue_t *queues, int niter)
{
    int tpwgts[2], limit, avgvwgt, origdiff;
    int mincutorder, newcut, mincut, initcut, mindiff;
    int pass, nswaps, from, to, vertex, begin, end, i, j, k, connect_partition, kwgt;

    tpwgts[0] = tvwgt * ntpwgts0;
    tpwgts[1] = tvwgt - tpwgts[0];
    limit = hunyuangraph_gpu_int_min(hunyuangraph_gpu_int_max(0.01 * nvtxs, 15), 100);
    avgvwgt = hunyuangraph_gpu_int_min((pwgts[0] + pwgts[1]) / 20, 2 * (pwgts[0] + pwgts[1]) / nvtxs);
    origdiff = hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]);

    for(pass = 0; pass < niter; pass++)
    {
        priority_queue_Reset(&queues[0], nvtxs);
        priority_queue_Reset(&queues[1], nvtxs);

        mincutorder = -1;
        newcut = mincut = initcut = edgecut;
        mindiff = hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]);

        for(i = 0;i < nvtxs;i++)
        {
            if(bnd[i] != 0)
                priority_queue_Insert(&queues[where[i]], i, ed[i] - id[i]);
        }

        for(nswaps = 0; nswaps < nvtxs; nswaps++)
        {
            from = (tpwgts[0] - pwgts[0] < tpwgts[1] - pwgts[1] ? 0 : 1);
            to = from ^ 1;

            vertex = priority_queue_GetTop(&queues[from]);
            if(vertex == -1)
                break;
            
            newcut -= (ed[vertex] - id[vertex]);

            if((newcut < mincut && hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]) <= origdiff + avgvwgt)
                || (newcut == mincut && hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]) < mindiff))
            {
                mincut  = newcut;
                mindiff = hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]);
                mincutorder = nswaps;
            }
            else if(nswaps - mincutorder > limit)
            { 
                newcut+=(ed[vertex] - id[vertex]);
                // pwgts[from] += vwgt[vertex];
                // pwgts[to]   -= vwgt[vertex];
                break;
            }

            begin = xadj[vertex];
            end   = xadj[vertex + 1];
            where[vertex] = to;
            pwgts[to]   += vwgt[vertex];
            pwgts[from] -= vwgt[vertex];
            moved[vertex] = nswaps;
            swaps[nswaps] = vertex;
            hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
            if(ed[vertex] == 0 && end != begin)
                bnd[vertex] = 0;
            
            for(j = begin; j < end; j++)
            {
                k = adjncy[j];
                connect_partition = where[k];
                if(to == connect_partition)
                    kwgt = adjwgt[j];
                else 
                    kwgt = -adjwgt[j];

                id[k] += kwgt;
                ed[k] -= kwgt;

                if(bnd[k] == 1)
                {
                    if(ed[k] == 0)
                    {
                        bnd[k] = 0;
                        if(moved[k] == -1)
                            priority_queue_Delete(&queues[connect_partition], k);
                    }
                    else
                    {
                        if(moved[k] == -1)
                            priority_queue_Update(&queues[connect_partition], k, ed[k] - id[k]);
                    }
                }
                else
                {
                    if(ed[k] > 0)
                    {
                        bnd[k] = 1;
                        if(moved[k] == -1)
                            priority_queue_Insert(&queues[connect_partition], k, ed[k] - id[k]);
                    }
                }
            }

        }

        for(i = 0;i < nswaps;i++)
            moved[swaps[i]] = -1;

        //  roll back        
        for(nswaps--;nswaps > mincutorder;nswaps--)
        {
            vertex = swaps[nswaps];
            from = where[vertex];
            to = from ^ 1;
            hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
            begin = xadj[vertex];
            end   = xadj[vertex + 1];
            if(ed[vertex] == 0 && bnd[vertex] == 1 && begin != end)
                bnd[vertex] = 0;
            else if(ed[vertex] > 0 && bnd[vertex] == 0)
                bnd[vertex] = 1;
            pwgts[to] += vwgt[vertex];
            pwgts[from] -= vwgt[vertex];

            for(j = begin; j < end;j++)
            {
                k = adjncy[j];
                connect_partition = where[k];
                if(to == connect_partition)
                    kwgt = adjwgt[j];
                else 
                    kwgt = -adjwgt[j];
                id[k] += kwgt;
                ed[k] -= kwgt;

                if(bnd[k] == 1 && ed[k] == 0)
                    bnd[k] = 0;
                if(bnd[k] == 0 && ed[k] > 0)
                    bnd[k] = 1;
            }
        }

        edgecut = mincut;

        if(mincutorder <= 0 || mincut == initcut)
            break;
    }

    //  free
}

__global__ void hunyuangraph_gpu_Bisection(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    int *global_edgecut, hunyuangraph_int8_t *global_where, priority_queue_t *queues, int *key, int *val, int *locator, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
	int ii = blockIdx.x;
    int tid = threadIdx.x;

    // printf("ii=%d tid=%d\n",ii, tid);

    // int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    
    int *queue = (int *)num;
    int *ed = queue + nvtxs;
    int *swaps = ed + nvtxs;
    int *tpwgts = (int *)(swaps + nvtxs);
    hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
    hunyuangraph_int8_t *moved = twhere + nvtxs;
    hunyuangraph_int8_t *bnd = moved + nvtxs;
    /*if(tid == 0)
    {
        printf("nvtxs=%d\n", nvtxs);
        printf("num=%p\n", num);
        printf("queue=%p\n", queue);
        printf("ed=%p\n", ed); 
        printf("swaps=%p\n", swaps);
        printf("tpwgts=%p\n", tpwgts);
        printf("twhere=%p\n", twhere);
        printf("moved=%p\n", moved);
        printf("bnd=%p\n", bnd);
    }*/
    // int *queue = (int *)(bnd + nvtxs);
    // int *ed = queue + nvtxs;
    // int *swaps = ed + nvtxs;
    // int *tpwgts = (int *)(swaps + nvtxs);

    for(int i = tid;i < nvtxs; i += blockDim.x)
    {
        twhere[i] = 1;
        moved[i] = 0;
    }
    if(tid == 0)
        tpwgts[0] = 0;
    else if(tid == 1)
        tpwgts[1] = tvwgt;
    __syncthreads();

    if(tid == 0)
    {
        int vertex, first, last, nleft, drain;
        int v, k, begin, end;

        vertex = ii;
        queue[0] = vertex;
        moved[queue[0]] = 1;
        first = 0;
        last = 1;
        nleft = nvtxs - 1;
        drain = 0;

        for(;;)
        {
            if (first == last) 
            {
                if (nleft == 0 || drain)
                    break;

                k = get_random_number_range(nleft, &state[ii]);
                // printf("inbfs=%"PRIDX" k=%"PRIDX"\n",inbfs,k);

                for (v = 0; v < nvtxs; v++) 
                {
                    if (moved[v] == 0) 
                    {
                        if (k == 0)
                            break;
                        else
                            k--;
                    }
                }

                queue[0] = v;
                moved[v] = 1;
                first    = 0; 
                last     = 1;
                nleft--;
            }
            
            v = queue[first];
            first++;
            // printf("v=%d\n", v);
            if(tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
            {
                drain = 1;
                continue;
            }

            twhere[v] = 0;
            tpwgts[0] += vwgt[v];
            tpwgts[1] -= vwgt[v];
            if(tpwgts[1] <= onemaxpwgt)
                break;
            
            drain = 0;
            end = xadj[v + 1];
            for(begin = xadj[v]; begin < end; begin++)
            {
                k = adjncy[begin];
                if(moved[k] == 0)
                {
                    queue[last] = k;
                    moved[k] = 1;
                    last++;
                    nleft--;
                }
            }
        }
    }
    // printf("tid=%d\n", tid);
    __syncthreads();

    // for(int i = tid;i < nvtxs;i += blockDim.x)
    //     global_where[i] = twhere[i];
    hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
    // if(ii == 1 && tid == 0)
    //     printf("ptr=%p\n", ptr);
    for(int i = tid;i < nvtxs; i += blockDim.x)
        ptr[i] = twhere[i];
    __syncthreads();
    // if(tid == 0)
    //     printf("gpu p=%d v=%d\n", p, ii);

    // printf("tid=%d\n", tid);
    
    //  compute ed, id, bnd
    int *id;
    id = queue;
    for(int i = tid;i < nvtxs; i += blockDim.x)
    {
        compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
    }
    __syncthreads();

    //  reduce ed to acquire the edgecut
    int edgecut;
        //  add to the first blockDim.x threads
    int *reduce_num = swaps;
    if(tid < nvtxs)
        reduce_num[tid] = ed[tid];
    for(int i = tid + blockDim.x;i < nvtxs; i += blockDim.x)
        reduce_num[tid] += ed[i];
    __syncthreads();

        //  if nvtxs < blockDim.x
    if(tid < nvtxs)
    {
        if(blockDim.x >= 512) 
        {
            if(tid < 256) reduce_num[tid] += reduce_num[tid + 256];
            __syncthreads();
        }
        if(blockDim.x >= 256) 
        {
            if(tid < 128) reduce_num[tid] += reduce_num[tid + 128];
            __syncthreads();
        }
        if(blockDim.x >= 128) 
        {
            if(tid < 64) reduce_num[tid] += reduce_num[tid + 64];
            __syncthreads();
        }

        if(tid < 32) warpReduction(reduce_num, tid, blockDim.x);

        if(tid == 0) edgecut = reduce_num[0];
    }
    // if(tid == 0)
    //     printf("gpu p=%d v=%d edgecut=%d\n", p, ii, edgecut);
    /*
    //  Balance the partition
        // if it is not necessary
    
    //  FM_2WayCutRefine
        //  serial
            //  init array moved
    for(int i = tid;i < nvtxs; i += blockDim.x)
        moved[i] = -1;
    if(tid == 0)
    {
        // allocate queues
        queues[ii * 2].key = &key[ii * 2];
        queues[ii * 2].val = &val[ii * 2];
        queues[ii * 2].locator = &locator[ii * 2];
        queues[ii * 2 + 1].key = &key[ii * 2 + 1];
        queues[ii * 2 + 1].val = &val[ii * 2 + 1];
        queues[ii * 2 + 1].locator = &locator[ii * 2 + 1];

        FM_2way_cut_refinement(nvtxs, vwgt, xadj, adjncy, adjwgt, twhere, tpwgts, ed, id, bnd, moved, swaps, tvwgt, edgecut, tpwgts0, &queues[ii * 2], 10);

        global_edgecut[ii] = edgecut;
    }

    hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
    for(int i = tid;i < nvtxs; i += blockDim.x)
        ptr[i] = twhere[i];
    
    if(tid == 0)
    {
        for(int i = 0;i < nvtxs;i++)
            printf("i=%d\n", i);
    }*/
}

__global__ void hunyuangraph_gpu_Bisection_global(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, hunyuangraph_int8_t *global_where, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
	int ii = blockIdx.x;
    int tid = threadIdx.x;

    // printf("ii=%d tid=%d\n",ii, tid);

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    // extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    num += ii * shared_size;
    int *queue = (int *)num;
    int *ed = queue + nvtxs;
    int *swaps = ed + nvtxs;
    int *tpwgts = (int *)(swaps + nvtxs);
    hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
    hunyuangraph_int8_t *moved = twhere + nvtxs;
    hunyuangraph_int8_t *bnd = moved + nvtxs;
    // if(tid == 0)
    //     printf("gpu i=%d where=%p\n", ii, twhere);
    /*if(tid == 0)
    {
        printf("nvtxs=%d\n", nvtxs);
        printf("num=%p\n", num);
        printf("queue=%p\n", queue);
        printf("ed=%p\n", ed); 
        printf("swaps=%p\n", swaps);
        printf("tpwgts=%p\n", tpwgts);
        printf("twhere=%p\n", twhere);
        printf("moved=%p\n", moved);
        printf("bnd=%p\n", bnd);
    }*/
    // int *queue = (int *)(bnd + nvtxs);
    // int *ed = queue + nvtxs;
    // int *swaps = ed + nvtxs;
    // int *tpwgts = (int *)(swaps + nvtxs);

    for(int i = tid;i < nvtxs; i += blockDim.x)
    {
        twhere[i] = 1;
        moved[i] = 0;
    }
    if(tid == 0)
        tpwgts[0] = 0;
    else if(tid == 1)
        tpwgts[1] = tvwgt;
    __syncthreads();

    if(tid == 0)
    {
        int vertex, first, last, nleft, drain;
        int v, k, begin, end;

        vertex = ii;
        queue[0] = vertex;
        moved[queue[0]] = 1;
        first = 0;
        last = 1;
        nleft = nvtxs - 1;
        drain = 0;

        for(;;)
        {
            if (first == last) 
            {
                if (nleft == 0 || drain)
                    break;

                k = get_random_number_range(nleft, &state[ii]);
                // printf("inbfs=%"PRIDX" k=%"PRIDX"\n",inbfs,k);

                for (v = 0; v < nvtxs; v++) 
                {
                    if (moved[v] == 0) 
                    {
                        if (k == 0)
                            break;
                        else
                            k--;
                    }
                }

                queue[0] = v;
                moved[v] = 1;
                first    = 0; 
                last     = 1;
                nleft--;
            }
            
            v = queue[first];
            first++;
            // printf("v=%d\n", v);
            if(tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
            {
                drain = 1;
                continue;
            }

            twhere[v] = 0;
            tpwgts[0] += vwgt[v];
            tpwgts[1] -= vwgt[v];
            if(tpwgts[1] <= onemaxpwgt)
                break;
            
            drain = 0;
            end = xadj[v + 1];
            for(begin = xadj[v]; begin < end; begin++)
            {
                k = adjncy[begin];
                if(moved[k] == 0)
                {
                    queue[last] = k;
                    moved[k] = 1;
                    last++;
                    nleft--;
                }
            }
        }
    }
    // printf("tid=%d\n", tid);
    __syncthreads();

    // for(int i = tid;i < nvtxs;i += blockDim.x)
    //     global_where[i] = twhere[i];
    /*hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
    // if(ii == 1 && tid == 0)
    //     printf("ptr=%p\n", ptr);
    for(int i = tid;i < nvtxs; i += blockDim.x)
        ptr[i] = twhere[i];
    __syncthreads();
    // if(tid == 0)
    //     printf("gpu p=%d v=%d\n", p, ii);

    // printf("tid=%d\n", tid);
    
    //  compute ed, id, bnd
    int *id;
    id = queue;
    for(int i = tid;i < nvtxs; i += blockDim.x)
    {
        compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
    }
    __syncthreads();

    //  reduce ed to acquire the edgecut
    int edgecut;
        //  add to the first blockDim.x threads
    // int *reduce_num = swaps;
    extern __shared__ int reduce_num[];
    if(tid < nvtxs)
        reduce_num[tid] = ed[tid];
    for(int i = tid + blockDim.x;i < nvtxs; i += blockDim.x)
        reduce_num[tid] += ed[i];
    __syncthreads();

        //  if nvtxs < blockDim.x
    if(tid < nvtxs)
    {
        if(blockDim.x >= 512) 
        {
            if(tid < 256) reduce_num[tid] += reduce_num[tid + 256];
            __syncthreads();
        }
        if(blockDim.x >= 256) 
        {
            if(tid < 128) reduce_num[tid] += reduce_num[tid + 128];
            __syncthreads();
        }
        if(blockDim.x >= 128) 
        {
            if(tid < 64) reduce_num[tid] += reduce_num[tid + 64];
            __syncthreads();
        }

        if(tid < 32) warpReduction(reduce_num, tid, blockDim.x);

        if(tid == 0) edgecut = reduce_num[0];
    }*/

    // if(tid == 0)
    //     printf("gpu p=%d v=%d edgecut=%d\n", p, ii, edgecut);
}

__global__ void hunyuangraph_gpu_Bisection_warp(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, hunyuangraph_int8_t *global_where, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + (tid >> 5);
    // int exam = blockIdx.x;
    // printf("p=%d exam=%d tid=%d ii=%d\n", p, exam, tid, ii);

    // printf("ii=%d tid=%d\n",ii, tid);

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    // extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    if(ii < nvtxs)
    {
        num += ii * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        hunyuangraph_int8_t *moved = twhere + nvtxs;
        hunyuangraph_int8_t *bnd = moved + nvtxs;
        // if(lane_id == 0)
        //     printf("gpu i=%d where=%p\n", ii, twhere);
        /*if(blockIdx.x == 0 && lane_id == 0)
        {
            printf("gpu i=%d nvtxs=%d\n", ii, nvtxs);
            printf("gpu i=%d num=%p\n", ii, num);
            printf("gpu i=%d queue=%p\n", ii, queue);
            printf("gpu i=%d ed=%p\n", ii, ed); 
            printf("gpu i=%d swaps=%p\n", ii, swaps);
            printf("gpu i=%d tpwgts=%p\n", ii, tpwgts);
            printf("gpu i=%d twhere=%p\n", ii, twhere);
            printf("gpu i=%d moved=%p\n", ii, moved);
            printf("gpu i=%d bnd=%p\n", ii, bnd);
        }*/
        // int *queue = (int *)(bnd + nvtxs);
        // int *ed = queue + nvtxs;
        // int *swaps = ed + nvtxs;
        // int *tpwgts = (int *)(swaps + nvtxs);

        for(int i = lane_id;i < nvtxs; i += 32)
        {
            twhere[i] = 1;
            moved[i] = 0;
        }
        if(lane_id == 0)
            tpwgts[0] = 0;
        else if(lane_id == 1)
            tpwgts[1] = tvwgt;
        __syncthreads();

        if(lane_id == 0)
        {
            int vertex, first, last, nleft, drain;
            int v, k, begin, end;

            vertex = ii;
            queue[0] = vertex;
            moved[queue[0]] = 1;
            first = 0;
            last = 1;
            nleft = nvtxs - 1;
            drain = 0;

            for(;;)
            {
                if (first == last) 
                {
                    if (nleft == 0 || drain)
                        break;

                    k = get_random_number_range(nleft, &state[ii]);
                    // printf("inbfs=%"PRIDX" k=%"PRIDX"\n",inbfs,k);

                    for (v = 0; v < nvtxs; v++) 
                    {
                        if (moved[v] == 0) 
                        {
                            if (k == 0)
                                break;
                            else
                                k--;
                        }
                    }

                    queue[0] = v;
                    moved[v] = 1;
                    first    = 0; 
                    last     = 1;
                    nleft--;
                }
                
                v = queue[first];
                first++;
                // printf("v=%d\n", v);
                if(tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
                {
                    drain = 1;
                    continue;
                }

                twhere[v] = 0;
                tpwgts[0] += vwgt[v];
                tpwgts[1] -= vwgt[v];
                if(tpwgts[1] <= onemaxpwgt)
                    break;
                
                drain = 0;
                end = xadj[v + 1];
                for(begin = xadj[v]; begin < end; begin++)
                {
                    k = adjncy[begin];
                    if(moved[k] == 0)
                    {
                        queue[last] = k;
                        moved[k] = 1;
                        last++;
                        nleft--;
                    }
                }
            }
        }
        // printf("tid=%d\n", tid);
        /*__syncthreads();

        // for(int i = tid;i < nvtxs;i += blockDim.x)
        //     global_where[i] = twhere[i];
        // hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
        // if(ii == 1 && tid == 0)
        //     printf("ptr=%p\n", ptr);
        // for(int i = lane_id;i < nvtxs; i += 32)
        //     ptr[i] = twhere[i];
        // __syncthreads();
        // if(tid == 0)
        //     printf("gpu p=%d v=%d\n", p, ii);

        // printf("tid=%d\n", tid);
        // if(lane_id == 0)
        // {
        //     for(int i = 0;i < nvtxs;i++)
        //         printf("i=%d where=%d\n",i, twhere[i]);
        //         printf("pu i=%d where=%p\n", ii, twhere);
        // }
        
        //  compute ed, id, bnd
        int *id;
        id = queue;
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
        }
        __syncthreads();

        //  reduce ed to acquire the edgecut
        int edgecut;
            //  add to the first blockDim.x threads
        // int *reduce_num = swaps;
        extern __shared__ int reduce_num[];
        if(lane_id < nvtxs)
            reduce_num[tid] = ed[lane_id];
        else 
            reduce_num[tid] = 0;
        for(int i = lane_id + 32;i < nvtxs; i += 32)
            reduce_num[tid] += ed[i];
        __syncthreads();

            //  if nvtxs < 32
        // if(lane_id < 32 && lane_id < nvtxs) 
        warpReduction(reduce_num, tid, 32);
        if(lane_id == 0) edgecut = reduce_num[(tid >> 5) * 32];

        // if(lane_id == 0)
        //     printf("gpu p=%d v=%d edgecut=%d\n", lane_id, ii, edgecut);*/
    }
}

__device__ bool can_moved(int *tpwgts, int vertex, int oneminpwgt, int *vwgt, hunyuangraph_int8_t *moved, int *drain)
{
    if(moved[vertex] > 1)
    {
        moved[vertex] = 1;
        return false;
    }
    if(tpwgts[0] > 0 && tpwgts[1] - vwgt[vertex] < oneminpwgt)
    {
        drain[0] = 1;
        return false;
        // continue;
        // return ;
    }
    return true;
}

__device__ bool is_balanced(int *tpwgts, int onemaxpwgt)
{
    if(tpwgts[1] <= onemaxpwgt)
        return true;
    else 
        return false;
}

__global__ void hunyuangraph_gpu_BFS_warp(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, hunyuangraph_int8_t *global_where, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + (tid >> 5);
    // int exam = blockIdx.x;
    // printf("p=%d exam=%d tid=%d ii=%d\n", p, exam, tid, ii);

    // printf("ii=%d tid=%d\n",ii, tid);

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    // extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    if(ii < nvtxs)
    {
        num += ii * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        hunyuangraph_int8_t *moved = twhere + nvtxs;
        hunyuangraph_int8_t *bnd = moved + nvtxs;
        // if(lane_id == 0)
        //     printf("gpu i=%d where=%p\n", ii, twhere);
        /*if(blockIdx.x == 0 && lane_id == 0)
        {
            printf("gpu i=%d nvtxs=%d\n", ii, nvtxs);
            printf("gpu i=%d num=%p\n", ii, num);
            printf("gpu i=%d queue=%p\n", ii, queue);
            printf("gpu i=%d ed=%p\n", ii, ed); 
            printf("gpu i=%d swaps=%p\n", ii, swaps);
            printf("gpu i=%d tpwgts=%p\n", ii, tpwgts);
            printf("gpu i=%d twhere=%p\n", ii, twhere);
            printf("gpu i=%d moved=%p\n", ii, moved);
            printf("gpu i=%d bnd=%p\n", ii, bnd);
        }*/
        // int *queue = (int *)(bnd + nvtxs);
        // int *ed = queue + nvtxs;
        // int *swaps = ed + nvtxs;
        // int *tpwgts = (int *)(swaps + nvtxs);

        for(int i = lane_id;i < nvtxs; i += 32)
        {
            twhere[i] = 1;
            moved[i] = 0;
        }
        if(lane_id == 0)
            tpwgts[0] = 0;
        else if(lane_id == 1)
            tpwgts[1] = tvwgt;
        __syncthreads();

        int vertex, first, last, nleft, drain;
        int v, k, begin, end, length, flag;
        
        if(lane_id == 0)
        {
            vertex = ii;
            queue[0] = vertex;
            first = 0;
            last = 1;
            drain = 0;
        }
        while(1)
        {
            __syncwarp();
            first  = __shfl_sync(0xffffffff, first, 0, 32);
            last   = __shfl_sync(0xffffffff, last, 0, 32);
            if(first == last)
                break;
            if(lane_id == 0)
            {
                vertex = queue[first];
                first++;
                flag = 0;

                if(!can_moved(tpwgts, vertex, oneminpwgt, vwgt, moved, &drain))
                    flag = 1;
            }
            // if(ii == 0 && lane_id == 0)
            //     printf("flag=%d\n", flag);
            __syncwarp();
            flag   = __shfl_sync(0xffffffff, flag, 0, 32);
            // if(ii == 0 && lane_id == 0)
            //     printf("flag=%d\n", flag);
            if(flag)
                continue;
            if(lane_id == 0)
            {
                twhere[vertex] = 0;
                tpwgts[0] += vwgt[vertex];
                tpwgts[1] -= vwgt[vertex];
            }
            // __syncwarp();
            // printf("p=%d 1207\n", p);
            __syncwarp();
            if(is_balanced(tpwgts, onemaxpwgt))
                break;
            
            vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
            begin = xadj[vertex];
            end   = xadj[vertex + 1];
            length = end - begin;
            first  = __shfl_sync(0xffffffff, first, 0, 32);
            last   = __shfl_sync(0xffffffff, last, 0, 32);
            last += length;
            // printf("p=%d 1218\n", p);
            //  push_queue
            for(int i = lane_id;i < length; i += 32)
            {
                k = adjncy[begin + i];
                queue[first + i] = k;
                moved[k]++;
            }

            // printf("p=%d 1227\n", p);
        }

        // printf("tid=%d\n", tid);
        __syncwarp();

        // for(int i = tid;i < nvtxs;i += blockDim.x)
        //     global_where[i] = twhere[i];
        // hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
        // if(ii == 1 && tid == 0)
        //     printf("ptr=%p\n", ptr);
        // for(int i = lane_id;i < nvtxs; i += 32)
        //     ptr[i] = twhere[i];
        // __syncthreads();
        // if(tid == 0)
        //     printf("gpu p=%d v=%d\n", p, ii);

        // printf("tid=%d\n", tid);
        // if(lane_id == 0)
        // {
        //     printf("ii=%d tpwgts[0]=%d tpwgts[1]=%d\n", ii, tpwgts[0], tpwgts[1]);
        //     for(int i = 0;i < nvtxs;i++)
        //         printf("i=%d where=%d\n",i, twhere[i]);
        // //         printf("pu i=%d where=%p\n", ii, twhere);
        // }
        
        //  compute ed, id, bnd
        /*int *id;
        id = queue;
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
        }
        __syncwarp();

        //  reduce ed to acquire the edgecut
        int edgecut;
            //  add to the first blockDim.x threads
        // int *reduce_num = swaps;
        extern __shared__ int reduce_num[];
        if(lane_id < nvtxs)
            reduce_num[tid] = ed[lane_id];
        else 
            reduce_num[tid] = 0;
        for(int i = lane_id + 32;i < nvtxs; i += 32)
            reduce_num[tid] += ed[i];
        __syncwarp();

            //  if nvtxs < 32
        // if(lane_id < 32 && lane_id < nvtxs) 
        warpReduction(reduce_num, tid, 32);
        if(lane_id == 0) edgecut = reduce_num[(tid >> 5) * 32];

        // if(lane_id == 0)
        //     printf("gpu p=%d v=%d edgecut=%d\n", lane_id, ii, edgecut);*/
    }
}


__device__ void warpGetMin(int *shared_edgecut, int *shared_id, int tid, int blocksize)
{
    if(blocksize >= 64)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 32])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 32];
            shared_id[tid] = shared_id[tid + 32];
        }
    }
    if(blocksize >= 32)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 16])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 16];
            shared_id[tid] = shared_id[tid + 16];
        }
    }
    if(blocksize >= 16)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 8])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 8];
            shared_id[tid] = shared_id[tid + 8];
        }
    }
    if(blocksize >= 8)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 4])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 4];
            shared_id[tid] = shared_id[tid + 4];
        }
    }
    if(blocksize >= 4)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 2])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 2];
            shared_id[tid] = shared_id[tid + 2];
        }
    }
    if(blocksize >= 2)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 1])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 1];
            shared_id[tid] = shared_id[tid + 1];
        }
    }
}

__global__ void hunyuangraph_gpu_select_where(int nvtxs, int *global_edgecut, int *best_id)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int num[2048];
    int *shared_edgecut, *shared_id;
    shared_edgecut = num;
    shared_id = shared_edgecut + blockDim.x;

    if(ii < nvtxs)
    {
        shared_edgecut[tid] = global_edgecut[ii];
        shared_id[tid] = ii;
    }
    __syncthreads();

    for(int i = tid + blockDim.x;i < nvtxs;i += blockDim.x)
    {
        if(shared_edgecut[tid] > shared_edgecut[i])
        {
            shared_edgecut[tid] = shared_edgecut[i];
            shared_id[tid] = shared_id[i];
        }
    }
    __syncthreads();

    if(tid < nvtxs)
    {
        if(blockDim.x >= 512) 
        {
            if(tid < 256) 
            {
                if(shared_edgecut[tid] > shared_edgecut[tid + 256])
                {
                    shared_edgecut[tid] = shared_edgecut[tid + 256];
                    shared_id[tid] = shared_id[tid + 256];
                }
            }
            __syncthreads();
        }
        if(blockDim.x >= 256) 
        {
            if(tid < 128) 
            {
                if(shared_edgecut[tid] > shared_edgecut[tid + 128])
                {
                    shared_edgecut[tid] = shared_edgecut[tid + 128];
                    shared_id[tid] = shared_id[tid + 128];
                }
            }
            __syncthreads();
        }
        if(blockDim.x >= 128) 
        {
            if(tid < 64) 
            {
                if(shared_edgecut[tid] > shared_edgecut[tid + 64])
                {
                    shared_edgecut[tid] = shared_edgecut[tid + 64];
                    shared_id[tid] = shared_id[tid + 64];
                }
            }
            __syncthreads();
        }

        if(tid < 32) 
            warpGetMin(shared_edgecut, shared_id, tid, blockDim.x);
        
        if(tid == 0)
            best_id[0] = shared_id[0];
    }
}

void hunyuangraph_gpu_RecursiveBisection(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int nparts, double *ubvec, double *tpwgts, int fpart)
{
    int *global_edgecut, *key, *val, *locator;
    hunyuangraph_int8_t *global_where;
    priority_queue_t *queues;
    int oneminpwgt, onemaxpwgt;
	double *tpwgts2, tpwgts0, tpwgts1;

	tpwgts2 = (double *)malloc(sizeof(double) * 2);
	tpwgts2[0] = hunyuangraph_double_sum(nparts>>1, tpwgts);
	tpwgts2[1] = 1.0 - tpwgts2[0];
    tpwgts0    = tpwgts2[0];
    // exit(0);
    
	//	GPU Bisection
    graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_where");
    // graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * 2, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_pwgts");
    // graph->cuda_ed    = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_ed");
    // graph->cuda_id    = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_id");
    // graph->cuda_bnd   = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_bnd");

    queues = (priority_queue_t *)lmalloc_with_check(sizeof(priority_queue_t) * graph->nvtxs * 2, "hunyuangraph_gpu_RecursiveBisection: queue");
    // key = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs * graph->nvtxs * 2, "hunyuangraph_gpu_RecursiveBisection: key");
    // val = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs * graph->nvtxs * 2, "hunyuangraph_gpu_RecursiveBisection: val");
    // locator = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs * graph->nvtxs * 2, "hunyuangraph_gpu_RecursiveBisection: locator");
    int shared_size = sizeof(hunyuangraph_int8_t) * graph->nvtxs * 3 + sizeof(int) * (graph->nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    hunyuangraph_int8_t *tnum;
    tnum = (hunyuangraph_int8_t *)lmalloc_with_check(shared_size * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: tnum");
    printf("tnum=%p\n", tnum);
    // exit(0);
    // for(int i = 0;i < graph->nvtxs * 2;i++)
    // {
    //     queues[i].key = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: queues[i].key");
    //     queues[i].val = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: queues[i].val");
    //     queues[i].locator = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: queues[i].locator");
    // }
    
    global_edgecut = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: global_edgecut");
    global_where   = (hunyuangraph_int8_t *)lmalloc_with_check(sizeof(hunyuangraph_int8_t) * graph->nvtxs * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: global_where");
    printf("global_where=%p\n", global_where);
    onemaxpwgt = ubvec[0] * graph->tvwgt[0] * tpwgts2[1];
    oneminpwgt = (1.0 / ubvec[0])*graph->tvwgt[0] * tpwgts2[1];

        //  CUDA Random number
    curandState *devStates;
    cudaMalloc(&devStates, graph->nvtxs * sizeof(curandState));
    initializeCurand<<<(graph->nvtxs + 127) / 128, 128>>>(-1, 0, graph->nvtxs, devStates);
    // curand_init(-1, 0, graph->nvtxs, devStates);

    // __global__ void hunyuangraph_gpu_Bisection(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    //     int *global_edgecut, int *global_where, priority_queue_t **queues, int oneminpwgt, int onemaxpwgt, curandState *state)
	// exit(0);
    printf("hunyuangraph_gpu_Bisection begin\n");
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_bisection, NULL);
    // hunyuangraph_gpu_Bisection<<<graph->nvtxs, 128, sizeof(hunyuangraph_int8_t) * graph->nvtxs * 3 + sizeof(int) * (graph->nvtxs * 3 + 2)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     global_edgecut, global_where, queues, key, val, locator, oneminpwgt, onemaxpwgt, devStates);
    // hunyuangraph_gpu_Bisection_global<<<graph->nvtxs, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     tnum, global_edgecut, global_where, oneminpwgt, onemaxpwgt, devStates);
    // hunyuangraph_gpu_Bisection_warp<<<(graph->nvtxs + 3) / 4, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     tnum, global_edgecut, global_where, oneminpwgt, onemaxpwgt, devStates);
    hunyuangraph_gpu_BFS_warp<<<(graph->nvtxs + 3) / 4, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
        tnum, global_edgecut, global_where, oneminpwgt, onemaxpwgt, devStates);
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_bisection, NULL);
    bisection_gpu_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;

    err = cudaDeviceSynchronize();
    checkCudaError(err, "hunyuangraph_gpu_Bisection");
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Kernel execution failed (error code %s)!\n", cudaGetErrorString(err));
    //     // Handle error as appropriate (e.g., exit, throw exception, etc.)
    // }
    printf("hunyuangraph_gpu_Bisection end\n");
    printf("        gpu_Bisection_time         %10.3lf %7.3lf%\n", bisection_gpu_time, bisection_gpu_time / bisection_gpu_time * 100);
    
    exit(0);

    for(int a = 0;a < graph->nvtxs;a++)
    {
        hunyuangraph_int8_t *num;
        num = (hunyuangraph_int8_t *)malloc(sizeof(hunyuangraph_int8_t) * graph->nvtxs);
        
        int *queue = (int *)tnum;
        int *ed = queue + graph->nvtxs;
        int *swaps = ed + graph->nvtxs;
        int *tpwgts = (int *)(swaps + graph->nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        twhere += a * shared_size;
        // printf("pu i=%d where=%p\n", a, twhere);
        cudaMemcpy(num, twhere, sizeof(hunyuangraph_int8_t) * graph->nvtxs, cudaMemcpyDeviceToHost);
        // cudaMemcpy(num, &global_where[a * graph->nvtxs], sizeof(hunyuangraph_int8_t) * graph->nvtxs, cudaMemcpyDeviceToHost);
        // if(a == 1)
        // {
        //     for(int i = 0;i < graph->nvtxs;i++)
        //         printf("i=%d where=%d\n", i, num[i]);
        // }
        int e = 0;
        for(int i = 0;i < graph->nvtxs;i++)
        {
            hunyuangraph_int8_t me = num[i];
            for(int j = graph->xadj[i];j < graph->xadj[i + 1];j++)
            {
                hunyuangraph_int8_t other = num[graph->adjncy[j]];
                if(other != me)
                    e += graph->adjwgt[j];
            }
        }

        printf("cpu v=%d edgecut=%d\n", a, e);
        free(num);
    }

    exit(0);

    //  select the best where
    int *best_id_gpu, best_id;
    best_id_gpu = (int *)lmalloc_with_check(sizeof(int), "best_id_gpu");
    hunyuangraph_gpu_select_where<<<(graph->nvtxs + 1023) / 1024, 1024, 2048 * sizeof(int)>>>(graph->nvtxs, global_edgecut, best_id_gpu);

    cudaMemcpy(&best_id, best_id_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(graph->cuda_where, &global_where[best_id * graph->nvtxs], sizeof(int) * graph->nvtxs, cudaMemcpyDeviceToDevice);

    // hunyuangraph_int8_t *answer = (hunyuangraph_int8_t *)malloc(sizeof(hunyuangraph_int8_t) * graph->nvtxs);

    cudaFree(devStates);

	//	update where
	// hunyuangraph_gpu_update_where(hunyuangraph_admin, graph, nparts, fpart);

	//	SplitGraph
	// hunyuangraph_splitgraph(hunyuangraph_admin, graph, &hunyuangraph_admin->lgraph, &hunyuangraph_admin->rgraph);

	//	free graph
	// rfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->where");

	//	Recursive
	if(nparts > 3)
	{
		// hunyuangraph_gpu_Recursivesisection(hunyuangraph_admin,rgraph,nparts-(nparts>>1),part,tpwgts+(nparts>>1),fpart+(nparts>>1),level);
	}
	else if(nparts == 3)
	{
		// hunyuangraph_free_graph(&lgraph);
		// hunyuangraph_gpu_RecursiveBisection(hunyuangraph_admin,rgraph,nparts-(nparts>>1),part,tpwgts+(nparts>>1),fpart+(nparts>>1),level);
	}

	free(tpwgts2);
}

void hunyuangraph_gpu_initialpartition(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	// allocate memory
	// GPU已分配 vwgt, xadj, adjncy, adjwgt
	// CPU已含有 nvtxs, nedges, tvwgt

	hunyuangraph_graph_t *t_graph = hunyuangraph_create_cpu_graph();

	t_graph->nvtxs = graph->nvtxs;
	t_graph->nedges = graph->nedges;
	t_graph->tvwgt = (int *)malloc(sizeof(int));
	t_graph->tvwgt[0] = graph->tvwgt[0];

	graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_initialpartition: graph->cuda_where");

	t_graph->cuda_vwgt   = (int *)lmalloc_with_check(sizeof(int) * t_graph->nvtxs, "hunyuangraph_gpu_initialpartition: t_graph->cuda_vwgt");
	t_graph->cuda_xadj   = (int *)lmalloc_with_check(sizeof(int) * (t_graph->nvtxs + 1), "hunyuangraph_gpu_initialpartition: t_graph->cuda_xadj");
	t_graph->cuda_adjncy = (int *)lmalloc_with_check(sizeof(int) * t_graph->nedges, "hunyuangraph_gpu_initialpartition: t_graph->cuda_adjncy");
	t_graph->cuda_adjwgt = (int *)lmalloc_with_check(sizeof(int) * t_graph->nedges, "hunyuangraph_gpu_initialpartition: t_graph->cuda_adjwgt");
	t_graph->cuda_label = (int *)lmalloc_with_check(sizeof(int) * t_graph->nvtxs, "hunyuangraph_gpu_initialpartition: t_graph->cuda_label");
    // t_graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * t_graph->nvtxs, "hunyuangraph_gpu_initialpartition: t_graph->cuda_where");

	cudaMemcpy(t_graph->cuda_vwgt, graph->cuda_vwgt, sizeof(int) * t_graph->nvtxs, cudaMemcpyDeviceToDevice);
	cudaMemcpy(t_graph->cuda_xadj, graph->cuda_xadj, sizeof(int) * (t_graph->nvtxs + 1), cudaMemcpyDeviceToDevice);
	cudaMemcpy(t_graph->cuda_adjncy, graph->cuda_adjncy, sizeof(int) * t_graph->nedges, cudaMemcpyDeviceToDevice);
	cudaMemcpy(t_graph->cuda_adjwgt, graph->cuda_adjwgt, sizeof(int) * t_graph->nedges, cudaMemcpyDeviceToDevice);

    t_graph->vwgt = (int *)malloc(sizeof(int) * t_graph->nvtxs);
    t_graph->xadj = (int *)malloc(sizeof(int) * (t_graph->nvtxs + 1));
    t_graph->adjncy = (int *)malloc(sizeof(int) * t_graph->nedges);
    t_graph->adjwgt = (int *)malloc(sizeof(int) * t_graph->nedges);

    memcpy(t_graph->vwgt, graph->vwgt, sizeof(int) * t_graph->nvtxs);
    memcpy(t_graph->xadj, graph->xadj, sizeof(int) * (t_graph->nvtxs + 1));
    memcpy(t_graph->adjncy, graph->adjncy, sizeof(int) * t_graph->nedges);
    memcpy(t_graph->adjwgt, graph->adjwgt, sizeof(int) * t_graph->nedges);



	double *ubvec, *tpwgts;
	ubvec  = (double *)malloc(sizeof(double));
	tpwgts = (double *)malloc(sizeof(double) * hunyuangraph_admin->nparts);
	
	ubvec[0] = (double)pow(hunyuangraph_admin->ubfactors[0], 1.0 / log(hunyuangraph_admin->nparts));
	for(int i = 0; i < hunyuangraph_admin->nparts; i++)
		tpwgts[i] = (double)hunyuangraph_admin->tpwgts[i];
    printf("hunyuangraph_gpu_RecursiveBisection begin\n");
	hunyuangraph_gpu_RecursiveBisection(hunyuangraph_admin, t_graph, hunyuangraph_admin->nparts, ubvec, tpwgts, 0);
    printf("hunyuangraph_gpu_RecursiveBisection end\n");
	free(ubvec);
	free(tpwgts);
}


#endif