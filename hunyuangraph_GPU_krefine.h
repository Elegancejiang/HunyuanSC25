#ifndef _H_KREFINE
#define _H_KREFINE

#include "hunyuangraph_struct.h"

/*CUDA-get the max/min pwgts*/
__global__ void Sum_maxmin_pwgts(int *maxwgt, int *minwgt, float *tpwgts, int tvwgt, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nparts)
	{
		float result = tpwgts[ii] * tvwgt;

		maxwgt[ii] = int(result * 1.03);
		minwgt[ii] = int(result / 1.03);
	}
}

/*CUDA-init boundary vertex num*/
__global__ void initbndnum(int *bndnum)
{
  bndnum[0] = 0;
}

/*CUDA-find vertex where ed-id>0 */
__global__ void Find_real_bnd_info(int *cuda_real_bnd_num, int *cuda_real_bnd, int *where, \
	int *xadj, int *adjncy, int *adjwgt, int nvtxs, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs) // && moved[ii] == 0
	{
		int i, k, begin, end, me, other, from;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		me    = 0;
		other = 0;
		from  = where[ii];

		for(i = begin;i < end;i++)
		{
			k = adjncy[i];
			if(where[k] == from) me += adjwgt[i];
			else other += adjwgt[i];
		}
		if(other > me) cuda_real_bnd[atomicAdd(&cuda_real_bnd_num[0],1)] = ii;
	}
}

/**/
__global__ void init_bnd_info(int *bnd_info, int length)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < length)
    bnd_info[ii] = 0;
}

/*CUDA-find boundary vertex should ro which part*/
__global__ void find_kayparams(int *cuda_real_bnd_num, int *bnd_info, int *cuda_real_bnd, int *where, \
int *xadj, int *adjncy, int *adjwgt, int nparts, int *cuda_bn, int *to, int *gain)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<cuda_real_bnd_num[0])
  {
    int pi, other, i, k, me_wgt, other_wgt;
    int start, end, begin, last;

    pi    = cuda_real_bnd[ii];
    begin = xadj[pi];
    last  = xadj[pi+1];
    start = nparts * ii;
    end   = nparts * (ii+1);
    other = where[pi];

    for(i = begin;i < last;i++)
    {
      k = adjncy[i];
      k = start + where[k];
      bnd_info[k] += adjwgt[i];
    }

    me_wgt = other_wgt = bnd_info[start + other];

    for(i=start;i<end;i++)
    {
      k = bnd_info[i];
      if(k > other_wgt)
      {
        other_wgt = k;
        other     = i - start;
      }
    }

    gain[ii]  = other_wgt - me_wgt;
    to[ii] = other;
    cuda_bn[ii] = pi;

  }
}

/*CUDA-init params*/
__global__ void initcucsr(int *cu_csr, int *bndnum)
{
  cu_csr[0] = 0;
  cu_csr[1] = bndnum[0];
}

/**/
__global__ void init_cu_que(int *cuda_que, int length)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < length)
    cuda_que[ii] = -1;
}

/*CUDA-get a csr array*/
__global__ void findcsr(int *to, int *cuda_que, int *bnd_num, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < nparts)
  {
    int i, t;
    int begin, end;
    begin = 2 * ii;
    end   = bnd_num[0];

    for(i = 0;i < end;i++)
    {
      if(ii == to[i])
      {
        cuda_que[begin] = i;
        break; 
      }
    }

    t = cuda_que[begin];

    if(t!=-1)
    {
      for(i = t;i < end;i++)
      {
        if(to[i] != ii)
        {
          cuda_que[begin + 1] = i - 1;
          break; 
        }
      }
    }

    t = 2 * to[end - 1] + 1;
    cuda_que[t] = end - 1;
  }
}

/*Find boundary vertex information*/
void hunyuangraph_findgraphbndinfo(hunyuangraph_admin_t *hunyuangraph_admin,hunyuangraph_graph_t *graph)
{
	int nvtxs  = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;
	int bnd_num; 

	initbndnum<<<1,1>>>(graph->cuda_bndnum);

	Find_real_bnd_info<<<nvtxs / 32 + 1,32>>>(graph->cuda_bndnum,graph->cuda_bnd,graph->cuda_where,\
		graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,nvtxs,nparts);
  
	cudaMemcpy(&bnd_num,graph->cuda_bndnum, sizeof(int), cudaMemcpyDeviceToHost);
  
	if(bnd_num > 0)
	{
		// cudaMalloc((void**)&graph->cuda_info, bnd_num * nparts * sizeof(int));
		if(GPU_Memory_Pool)
			graph->cuda_info = (int *)rmalloc_with_check(sizeof(int) * bnd_num * nparts,"info");
		else
			cudaMalloc((void **)&graph->cuda_info, sizeof(int) * bnd_num * nparts);

		init_bnd_info<<<bnd_num * nparts / 32 + 1,32>>>(graph->cuda_info, bnd_num * nparts);

		find_kayparams<<<bnd_num/32+1,32>>>(graph->cuda_bndnum,graph->cuda_info,graph->cuda_bnd,graph->cuda_where,\
			graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,nparts,graph->cuda_bn,graph->cuda_to,graph->cuda_gain);

		initcucsr<<<1,1>>>(graph->cuda_csr,graph->cuda_bndnum);

		bb_segsort(graph->cuda_to, graph->cuda_bn, bnd_num, graph->cuda_csr, 1);
		
		init_cu_que<<<2 * nparts / 32 + 1,32>>>(graph->cuda_que, 2 * nparts);
		
		findcsr<<<nparts/32+1,32>>>(graph->cuda_to,graph->cuda_que,graph->cuda_bndnum,nparts);
		
		if(GPU_Memory_Pool)
			rfree_with_check((void *)graph->cuda_info, sizeof(int) * bnd_num * nparts,"graph->cuda_info");	//	graph->cuda_info
		else
			cudaFree(graph->cuda_info);
	}

	graph->cpu_bndnum=(int *)malloc(sizeof(int));
	graph->cpu_bndnum[0]=bnd_num;
}

/*CUDA-move vertex*/
__global__ void Exnode_part1(int *cuda_que, int *pwgts, int *bnd, int *bndto, int *vwgt,\
  int *maxvwgt, int *minvwgt, int *where, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<nparts)
  {
    int i, me, to, vvwgt;
    int memax, memin, tomax, tomin, mepwgts, topwgts;
    int begin, end, k;
    begin = cuda_que[2 * ii];
    end   = cuda_que[2 * ii + 1];
    if(begin != -1)
    {
      for(i = begin;i <= end;i++)
      {
        k     = bnd[i];
        vvwgt = vwgt[k];
        me    = where[k];
        to    = bndto[i];

        memax   = maxvwgt[me];
        memin   = minvwgt[me];
        tomax   = maxvwgt[to];
        tomin   = minvwgt[to];
        mepwgts = pwgts[me];
        topwgts = pwgts[to];

        if(me <= to)
        {
          if(((topwgts + vvwgt >= tomin) && (topwgts + vvwgt <= tomax))\
            &&((mepwgts - vvwgt >= memin) && (mepwgts - vvwgt <= memax)))
          {
            atomicAdd(&pwgts[to],vvwgt);
            atomicSub(&pwgts[me],vvwgt);
            where[k] = to;
          }
        }
      }
    }
  }
}

/*CUDA-move vertex*/
__global__ void Exnode_part2(int *cuda_que, int *pwgts, int *bnd, int *bndto, int *vwgt,\
  int *maxvwgt, int *minvwgt, int *where, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<nparts)
  {
    int i, me, to, vvwgt;
    int memax, memin, tomax, tomin, mepwgts, topwgts;
    int begin, end, k;
    begin = cuda_que[2 * ii];
    end   = cuda_que[2 * ii + 1];
    if(begin != -1)
    {
      for(i = begin;i <= end;i++)
      {
        k     = bnd[i];
        vvwgt = vwgt[k];
        me    = where[k];
        to    = bndto[i];

        memax   = maxvwgt[me];
        memin   = minvwgt[me];
        tomax   = maxvwgt[to];
        tomin   = minvwgt[to];
        mepwgts = pwgts[me];
        topwgts = pwgts[to];

        if(me > to)
        {
          if(((topwgts+vvwgt>=tomin)&&(topwgts+vvwgt<=tomax))\
          &&((mepwgts-vvwgt>=memin)&&(mepwgts-vvwgt<=memax)))
          {
            atomicAdd(&pwgts[to],vvwgt);
            atomicSub(&pwgts[me],vvwgt);
            where[k] = to;
          }
        }
      }
    }
  }
}

/*Graph multilevel uncoarsening algorithm*/
void hunyuangraph_k_refinement(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs  = graph->nvtxs;

	Sum_maxmin_pwgts<<<nparts / 32 + 1,32>>>(graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_tpwgts,graph->tvwgt[0],nparts);

	for(int i = 0;i < 2;i++)
	{
		hunyuangraph_findgraphbndinfo(hunyuangraph_admin,graph);

		if(graph->cpu_bndnum[0] > 0)
		{
			Exnode_part1<<<nparts/32+1,32>>>(graph->cuda_que,graph->cuda_pwgts,graph->cuda_bn,graph->cuda_to,graph->cuda_vwgt,\
				graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_where,nparts);

			Exnode_part2<<<nparts/32+1,32>>>(graph->cuda_que,graph->cuda_pwgts,graph->cuda_bn,graph->cuda_to,graph->cuda_vwgt,\
				graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_where,nparts);
		}
		else
			break;
	}
}

__global__ void init_connection(int nedges, int *connection, int *connection_to)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nedges)
	{
		connection[ii]    = 0;
		connection_to[ii] = -1;
	}
}

__global__ void select_bnd_vertices_old(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *connection, int *connection_to, int *gain, int *to)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int k, begin, end, length, where_i, where_k, hash_addr;
		char flag_bnd;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		length = end - begin;
		where_i = where[ii];
		flag_bnd = 0;

		for(int j = begin; j < end; j++)
		{
			k = adjncy[j];
			where_k = where[k];

			if(where_i != where_k)
				flag_bnd = 1;

			//	compute connection with other partitions for every vertex by hash table
			hash_addr = where_k & length;	//	hash value
			while(1)
			{
				int key_exist = connection_to[begin + hash_addr];
				if(key_exist == ii)
				{
					connection[begin + hash_addr] += adjwgt[j];
					break;
				}
				else if(key_exist == -1)
				{
					connection_to[begin + hash_addr] = where_k;
					connection[begin + hash_addr] += adjwgt[j];
				}
				else 
				{
					hash_addr = (hash_addr + 1) & length;
				}
			}

		}

	}
}

// 在warp内计算最大值，返回lane 0线程的值
__device__ int warpReduceMax(int val, int to, int lane_id, int ii) 
{
    int tmp_val;

	if(ii == 49913 && lane_id < 32)
		printf("32 ii:%d lane_id:%d %d %d\n",ii, lane_id, val, to);
	tmp_val = __shfl_down_sync(0xffffffff, val, 16);
	if(tmp_val > val)
	{
		val = tmp_val;
		to = __shfl_down_sync(0xffffffff, to, 16);
	}
	if(ii == 49913 && lane_id < 16)
		printf("16 ii:%d lane_id:%d %d %d\n",ii, lane_id, val, to);
	tmp_val = __shfl_down_sync(0xffffffff, val, 8);
	if(tmp_val > val)
	{
		val = tmp_val;
		to = __shfl_down_sync(0xffffffff, to, 8);
	}
	if(ii == 49913 && lane_id < 8)
		printf("8  ii:%d lane_id:%d %d %d\n",ii, lane_id, val, to);
	tmp_val = __shfl_down_sync(0xffffffff, val, 4);
	if(tmp_val > val)
	{
		val = tmp_val;
		to = __shfl_down_sync(0xffffffff, to, 4);
	}
	if(ii == 49913 && lane_id < 4)
		printf("4  ii:%d lane_id:%d %d %d\n",ii, lane_id, val, to);
	tmp_val = __shfl_down_sync(0xffffffff, val, 2);
	if(tmp_val > val)
	{
		val = tmp_val;
		to = __shfl_down_sync(0xffffffff, to, 2);
	}
	if(ii == 49913 && lane_id < 2)
		printf("2  ii:%d lane_id:%d %d %d\n",ii, lane_id, val, to);
	tmp_val = __shfl_down_sync(0xffffffff, val, 1);
	if(tmp_val > val)
	{
		val = tmp_val;
		to = __shfl_down_sync(0xffffffff, to, 1);
	}
	if(ii == 49913 && lane_id < 1)
		printf("1  ii:%d lane_id:%d %d %d\n",ii, lane_id, val, to);

    return to;
}

__global__ void select_bnd_vertices_warp(int nvtxs, int nparts, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *gain, int *to)
{
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + warp_id;

	extern __shared__ int connection_shared[];
	int *ptr_shared = connection_shared + warp_id * nparts;
	int *to_shared = ptr_shared + 4 * nparts;
	int *id = connection_shared + 8 * nparts;
	int *flag_bnd = id + 4;

	// if(ii == 5920 && lane_id == 0)
	// {
	// 	// printf("ii:%d lane_id:%d\n", ii, lane_id);
	// 	printf("ii:%d lane_id:%d id: %p, flag_bnd: %p, ptr_shared: %p, to_shared: %p\n", ii, lane_id, id, flag_bnd, ptr_shared, to_shared);
	// }

	for(int i = lane_id;i < nparts;i += 32)
	{
		ptr_shared[i] = 0;
		to_shared[i] = i;
		id[warp_id] = 0;
		flag_bnd[warp_id] = 0;
	}
	__syncwarp();
	// __syncthreads();

	if(ii < nvtxs)
	{
		int begin, end, where_i, where_k, j, k;
		begin   = xadj[ii];
		end     = xadj[ii + 1];
		where_i = where[ii];
		ptr_shared[where_i] = -1;

		for(j = begin + lane_id;j < end;j += 32)
		{
			k = adjncy[j];
			where_k = where[k];

			if(where_i != where_k)
			{
				// printf("flag_bnd\n");
				flag_bnd[warp_id] = 1;
				atomicAdd(&ptr_shared[where_k], adjwgt[j]);
			}
			else	
				atomicAdd(&id[warp_id], adjwgt[j]);
		}
		__syncwarp();
		// __syncthreads();

		//	bnd vertex
		if(flag_bnd[warp_id] == 1)
		{
			end = nparts / 2;
			while(end > 32)
			{
				for(j = lane_id;j < end;j += 32)
				{
					if(ptr_shared[j + end] > ptr_shared[j])
					{
						ptr_shared[j] = ptr_shared[j + end];
						to_shared[j] = to_shared[j + end];
					}
				}
				end >>= 1;
				__syncwarp();
				// __syncthreads();
			}
			// if(lane_id >= end)
			// {
			// 	ptr_shared[lane_id] = -1;
			// 	to_shared[lane_id] = -1;
			// }
			// // __syncwarp();
			// __syncthreads();
			// else
			// {
			// 	k = ptr_shared[lane_id];
			// 	// j = to_shared[lane_id];
			// 	j = lane_id;
			// }
			// if(ii == 5920 && lane_id == 0)
			// 	printf("i:8 ii: %d, where_i:%d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
			// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
			
			// __syncwarp();
			// __syncthreads();
			for(int i = end; i > 0; i >>= 1)
			{
				// if(lane_id == 0)
				// 	printf("ii: %d i: %d\n", ii, i);
				if(lane_id < i && ptr_shared[lane_id + i] > ptr_shared[lane_id])
				{
					ptr_shared[lane_id] = ptr_shared[lane_id + i];
					to_shared[lane_id] = to_shared[lane_id + i];
				}
				__syncwarp();
				// __syncthreads();
				// if(ii == 5920 && lane_id == 0)
				// 	printf("i:%d ii: %d, where_i:%d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", i, ii, where_i, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// __syncwarp();
				// __syncthreads();
			}
			
			// if(lane_id == 0)
			// {
			// 	printf("ii: %d, where_i:%d  %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
			// 	ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
			// }
			// __syncwarp();
			// j = warpReduceMax(k, j, lane_id, ii);

			if(lane_id == 0)
			{
				j = to_shared[0];
				// if(j == -1)
				// {
				// 	printf("ii:%d warp_id:%d lane_id:%d id: %p, flag_bnd: %p, ptr_shared: %p, to_shared: %p\n", ii, warp_id, lane_id, id, flag_bnd, ptr_shared, to_shared);
	
				// 	printf("ii: %d, where_i:%d j:%d to:%d k: %d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, j, to_shared[j], k, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// }
				k = ptr_shared[j] - id[warp_id];
				// printf("ii: %d, where_i:%d k: %d id: %d\n", ii, where_i, k, id[warp_id]);
				// if(ii == 473678)
				// 	printf("ii: %d, where_i:%d j:%d to:%d k: %d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, j, to_shared[j], k, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// if(k >= -0.50 * id[warp_id] / sqrtf(begin - end))
				if(k >= -0.15 * id[warp_id])
				{
					to[ii] = j;
					gain[ii] = k;
					select[ii] = 1;
					// printf("ii: %d\n", ii);
				}
				else 
					select[ii] = 0;
				// if(j == where_i)
				// 	printf("ii: %d, where_i: %d to_i: %d select: %d\n", ii, where_i, j, (int)select[ii]);
			}
		}
		else 
		{
			if(lane_id == 0)
				select[ii] = 0;
		}
	}
}

__global__ void moving_vertices_interaction_SC25(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *gain, int *to)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && select[ii] == 1)
	{
		int begin, end, k, where_i, where_k, to_i, to_k, interaction_i, interaction_k, edgewgt;
		where_i = where[ii];
		to_i    = to[ii];
		if(where_i != to_i)
		{
			begin   = xadj[ii];
			end     = xadj[ii + 1];

			interaction_i = 0;
			for(int j = begin;j < end;j++)
			{
				k = adjncy[j];
				if(select[k] == 1 && where_k != to_k)
				{
					where_k = where[k];
					to_k = to[k];
					interaction_k = 0;
					edgewgt = adjwgt[j];
					if(where_k == where_i && to_k == to_i)
					{
						// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
						interaction_i += edgewgt;
						interaction_k += edgewgt;
					}
					else if(where_k == where_i && to_k != to_i)
					{
						// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
						interaction_i += edgewgt;
					}
					else if(where_k != where_i && to_k == to_i)
					{
						// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
						interaction_k += edgewgt;
					}
					else if(where_k != where_i && to_k != to_i)
					{
						if(where_k != where_i && where_k != to_i && to_k != where_i)
						{
							// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
							// inter_action[3] += graph->adjwgt[j];
						}
						else if(to_k == where_i && to_i != where_k)
						{
							// if(where_i != to_k || to_i != where_k)
							// 	printf("ATTENTION where_i=%"PRIDX" to_k=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX"\n", where_i, to_k, to_i, where_k);
							// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
							interaction_k -= edgewgt;
						}
						else if(where_k == to_i && to_k != where_i)
						{
							// if(where_i != to_k || to_i != where_k)
							// 	printf("ATTENTION where_i=%"PRIDX" to_k=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX"\n", where_i, to_k, to_i, where_k);
							// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
							interaction_i -= edgewgt;
						}
						else if(where_k == to_i && to_k == where_i)
						{
							// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
							interaction_i -= edgewgt;
							interaction_k -= edgewgt;
						}
					}
					atomicAdd(&gain[k], interaction_k);
				}
			}
			atomicAdd(&gain[ii], interaction_i);
		}
	}
}

__global__ void update_select_SC25(int nvtxs, char *select, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && select[ii] == 1 && gain[ii] < 0)
	{
		select[ii] = 0;
	}
}

__global__ void execute_move(int nvtxs, int *vwgt, int *where, int *pwgts, char *select, int *to)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && select[ii] == 1)
	{
		int t = to[ii];
		int vwgt_i = vwgt[ii];
		int where_i = where[ii];
		atomicAdd(&pwgts[t], vwgt_i);
		where[ii] = t;
		atomicSub(&pwgts[where_i], vwgt_i);
	}
}

__global__ void execute_move_balance(int nvtxs, int *vwgt, int *where, int *pwgts, int *maxwgt, char *select, int *to)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && select[ii] == 1)
	{
		int t = to[ii];
		int vwgt_i = vwgt[ii];
		if(pwgts[t] + vwgt_i < maxwgt[t])
		{
			int where_i = where[ii];
			atomicAdd(&pwgts[t], vwgt_i);
			where[ii] = t;
			atomicSub(&pwgts[where_i], vwgt_i);
		}
	}
}

__global__ void exam_where(int nvtxs, int *where)
{
	for(int i = 0;i < nvtxs && i < 100;i++)
		printf("%5d\n", where[i]);
}

__global__ void is_balance(int nvtxs, int nparts, int *pwgts, int *maxwgt, int *balance)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nparts)
	{
		printf("ii:%d pwgts:%d maxwgt:%d\n", ii, pwgts[ii], maxwgt[ii]);
		int flag = (pwgts[ii] > maxwgt[ii]);
		if(!flag)
		{
			balance[0] = 1;
			// printf("balance!\n");
		}
		else 
		{
			balance[0] = 0;
			// printf("imbalance!\n");
		}
	}
}

void hunyuangraph_k_refinement_SC25(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	// printf("hunyuangraph_k_refinement_SC25 begin\n");
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs  = graph->nvtxs;
	int nedges = graph->nedges;

	int edgecut;
	int *balance = (int *)malloc(sizeof(int));

	Sum_maxmin_pwgts<<<nparts / 32 + 1,32>>>(graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_tpwgts,graph->tvwgt[0],nparts);
	// cudaDeviceSynchronize();
	// exam_balance<<<1,1>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_minwgt);
	// cudaDeviceSynchronize();
	for(int iter = 0;iter < 5;iter++)
	{
		// init_connection<<<(nedges + 127) / 128, 128>>>(nedges, graph->cuda_connection, graph->cuda_connection_to);
		// cudaDeviceSynchronize();
		// exam_where<<<1, 1>>>(nvtxs, graph->cuda_where);
		// cudaDeviceSynchronize();

		// cudaDeviceSynchronize();
		// is_balance<<<nparts, 32>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_balance);
		// cudaDeviceSynchronize();

		cudaMemcpy(balance, graph->cuda_balance, sizeof(int), cudaMemcpyDeviceToHost);
		// printf("balance=%d\n", balance[0]);

		//	bnd and gain >= -0.15 * id
		gettimeofday(&begin_general, NULL);
		cudaDeviceSynchronize();
		select_bnd_vertices_warp<<<(nvtxs + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(nvtxs, nparts, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, \
			graph->cuda_select, graph->cuda_gain, graph->cuda_to);
		cudaDeviceSynchronize();
		gettimeofday(&end_general, NULL);
		uncoarsen_select_bnd_vertices_warp += (end_general.tv_sec - begin_general.tv_sec) * 1000.0 + (end_general.tv_usec - begin_general.tv_usec) / 1000.0;
		// printf("select_bnd_vertices_warp end %10.3lf\n", uncoarsen_select_bnd_vertices_warp);
		// cudaDeviceSynchronize();
		// exam_where<<<1, 1>>>(nvtxs, graph->cuda_where);
		// cudaDeviceSynchronize();
		
		// cudaDeviceSynchronize();
		// int select_sum = compute_graph_select_gpu(graph);
		// cudaDeviceSynchronize();
		// printf("second_select=%10d\n", select_sum);

		//	update select
		cudaDeviceSynchronize();
		moving_vertices_interaction_SC25<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, \
			graph->cuda_select, graph->cuda_gain, graph->cuda_to);
		cudaDeviceSynchronize();
		update_select_SC25<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_select, graph->cuda_gain);

		// cudaDeviceSynchronize();
		// select_sum = compute_graph_select_gpu(graph);
		// cudaDeviceSynchronize();
		// printf("second_select=%10d\n", select_sum);

		execute_move<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_vwgt, graph->cuda_where, graph->cuda_pwgts, graph->cuda_select, graph->cuda_to);
		// execute_move<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_vwgt, graph->cuda_where, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_select, graph->cuda_to);

		// cudaDeviceSynchronize();
		// exam_where<<<1, 1>>>(nvtxs, graph->cuda_where);
		// cudaDeviceSynchronize();

		// cudaDeviceSynchronize();
		// compute_edgecut_gpu(graph->nvtxs, &edgecut, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);
		// cudaDeviceSynchronize();
		// printf("iter:%d %10d\n", iter, edgecut);

		// printf("iter:%d %10d\n", iter, edgecut);
		// cudaDeviceSynchronize();
		// is_balance<<<nparts, 32>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_balance);
		// cudaDeviceSynchronize();
	}
}

#endif