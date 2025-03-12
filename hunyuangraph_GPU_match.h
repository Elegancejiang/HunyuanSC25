#ifndef _H_GPU_MATCH
#define _H_GPU_MATCH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_GPU_prefixsum.h"
#include "hunyuangraph_bb_segsort.h"
#include "bb_segsort.h"

/*CUDA-init match array*/
__global__ void init_gpu_match(int *match, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
		match[ii] = -1;
}

/*CUDA-hem matching*/
__global__ void cuda_hem(int nvtxs_hem, int *match, int *xadj, int *vwgt, int *adjwgt, int *adjncy, int maxvwgt_hem)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	/*int maxvwgt, maxidx, maxwgt, i, j, ivwgt, k, jw;
	int ibegin, iend, begin, end;

	maxvwgt = maxvwgt_hem;

	if(ii < addition) ibegin = ii * (size + 1), iend = ibegin + size + 1;
	else ibegin = addition * (size + 1) + (ii - addition) * size, iend = ibegin + size;

	for(i = ibegin;i < iend;i++)
	{
		if(match[i] == -1)
		{
			begin = xadj[i];
			end   = xadj[i + 1];
			ivwgt = vwgt[i];

			maxidx = i;
			maxwgt = -1;

			for(j = begin;j < end;j++)
			{
				k  = adjncy[j];
				jw = adjwgt[j];
				if(match[k] == -1 && maxwgt < jw && ivwgt + vwgt[k] <= maxvwgt)
				{
					maxidx = k;
					maxwgt = jw;
				}
			}
				if(maxidx == i && 3 * ivwgt < maxvwgt)
					maxidx = -1;

			if(maxidx != -1)
			{
				match[i] = maxidx;
				atomicExch(&match[maxidx],i);
			}
		}
	}*/

	int tt, nvtxs, maxvwgt, b, a, x, maxidx, maxwgt, i, j, ivwgt, k, jw;
	int ibegin, iend, begin, end;

	tt = 1024;
	nvtxs = nvtxs_hem;
	maxvwgt = maxvwgt_hem;

	if (nvtxs % tt == 0)
	{
		b = nvtxs / tt;
		ibegin = ii * b;
		iend = ibegin + b;
	}
	else
	{
		b = nvtxs / tt;
		a = b + 1;
		x = nvtxs - b * tt;
		if (ii < x)
		{
			ibegin = ii * a;
			iend = ibegin + a;
		}
		else
		{
			ibegin = ii * b + x;
			iend = ibegin + b;
		}
	}
	for (i = ibegin; i < iend; i++)
	{
		if (match[i] == -1)
		{
			begin = xadj[i];
			end = xadj[i + 1];
			ivwgt = vwgt[i];

			maxidx = i;
			maxwgt = -1;

			if (ivwgt < maxvwgt)
			{
				for (j = begin; j < end; j++)
				{
					k = adjncy[j];
					jw = adjwgt[j];
					if (match[k] == -1 && maxwgt < jw && ivwgt + vwgt[k] <= maxvwgt)
					{
						maxidx = k;
						maxwgt = jw;
					}
				}
				if (maxidx == i && 3 * ivwgt < maxvwgt)
					maxidx = -1;
			}

			if (maxidx != -1)
			{
				atomicCAS(&match[maxidx], -1, i);
				atomicExch(&match[i], maxidx);
			}
		}
	}
}

__global__ void cuda_hem_test(int nvtxs_hem, int *match, int *xadj, int *vwgt, int *adjwgt, int *adjncy, int maxvwgt_hem)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs_hem)
	{
		if (ii % 2 == 0)
			match[ii] = ii + 1;
		else
			match[ii] = ii - 1;

		if (ii == nvtxs_hem - 1 && ii % 2 == 0)
			match[ii] = ii;
	}
}

__global__ void cuda_hem_229_3(int nvtxs, int *match, int *xadj, int *vwgt, int *adjwgt, int *adjncy, int maxvwgt)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int maxidx, maxwgt, j, k, ivwgt, jw;
		int begin, end;
		begin = xadj[ii];
		end = xadj[ii + 1];
		ivwgt = vwgt[ii];

		if (match[ii] == -1)
		{
			begin = xadj[ii];
			end = xadj[ii + 1];
			ivwgt = vwgt[ii];

			maxidx = ii;
			maxwgt = -1;

			if (ivwgt < maxvwgt)
			{
				for (j = begin; j < end; j++)
				{
					k = adjncy[j];
					jw = adjwgt[j];
					if (match[k] == -1 && maxwgt < jw && ivwgt + vwgt[k] <= maxvwgt)
					{
						maxidx = k;
						maxwgt = jw;
					}
				}
				if (maxidx == ii && 3 * ivwgt < maxvwgt)
					maxidx = -1;
			}

			if (maxidx != -1)
			{
				atomicCAS(&match[maxidx], -1, ii);
				atomicExch(&match[ii], maxidx);
			}
		}
	}
}

__global__ void reset_match(int nvtxs, int *match)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int t = atomicAdd(&match[ii], 0);
		if (t != -1)
		{
			if (match[t] != ii)
				atomicExch(&match[ii], -1);
		}
	}
}

__global__ void random_match(int nvtxs, int *xadj, int *adjncy, int *match)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		if(match[ii] == -1)
		{
			int begin, end, i, j, jw, maxidx, maxwgt;
			begin = xadj[ii];
			end = xadj[ii + 1];
			maxidx = ii;
			maxwgt = -1;

			for (i = begin;i < end;i++)
			{
				j  = adjncy[i];
				if (match[j] == -1)
				{
					jw = xadj[j + 1] - xadj[j];
					if(ii < 200)
						printf("ii=%d j=%d jw=%d\n",ii, j, jw);
					if(maxwgt < jw)
					{
						maxwgt = jw;
						maxidx = j;
						if(ii < 200)
							printf("ii=%d j=%d jw=%d maxidx=%d\n",ii, j, jw, maxidx);
					}
				}
			}

			if (maxidx != ii)
			{
				atomicCAS(&match[maxidx], -1, ii);
				atomicExch(&match[ii], maxidx);
			}
		}
	}
}

__global__ void random_match_group(int nvtxs, int *xadj, int *adjncy, int *match)
{
	int ii = blockIdx.x;

	int tt, b, a, x, maxidx, maxwgt, i, j, ivwgt, k, jw;
	int ibegin, iend, begin, end;

	tt = 1024;

	if (nvtxs % tt == 0)
	{
		b = nvtxs / tt;
		ibegin = ii * b;
		iend = ibegin + b;
	}
	else
	{
		b = nvtxs / tt;
		a = b + 1;
		x = nvtxs - b * tt;
		if (ii < x)
		{
			ibegin = ii * a;
			iend = ibegin + a;
		}
		else
		{
			ibegin = ii * b + x;
			iend = ibegin + b;
		}
	}

	// if(ii == 0)
	// 	printf("ii=%d ibegin=%d iend=%d\n",ii, ibegin, iend);

	for (i = ibegin; i < iend; i++)
	{
		// if(ii == 0)
		// 	printf("i=%d match=%d\n", i, match[i]);
		if (match[i] == -1)
		{
			begin = xadj[i];
			end = xadj[i + 1];

			maxidx = i;
			maxwgt = -1;

			for (j = begin; j < end; j++)
			{
				k = adjncy[j];
				if (match[k] == -1)
				{
					jw = xadj[k + 1] - xadj[k];
					// if(ii == 0)
					// 	printf("ii=%d i=%d k=%d jw=%d\n",ii, i, k, jw);
					if(maxwgt < jw)
					{
						maxidx = k;
						maxwgt = jw;
						// if(ii == 0)
						// 	printf("ii=%d i=%d k=%d jw=%d maxidx=%d\n",ii, i, k, jw, maxidx);
					}
				}
			}
		}

		if (maxidx != -1)
		{
			if(atomicCAS(&match[maxidx], -1, i) == -1)
				atomicExch(&match[i], maxidx);
		}
	}
}

__global__ void init_gpu_receive_send(int *num, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
		num[ii] = -1;
}

__global__ void set_receive_send(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *match, int *receive, int *send, int offset)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs && match[ii] == -1)
	{
		int i, j, k, kk;
		int begin, end;
		begin = xadj[ii];
		end = xadj[ii + 1];

		for (i = end - 1, k = 0; i >= begin && k < offset; i--)
		{
			j = adjncy[i];
			if (match[j] == -1)
			{
				send[ii * offset + k] = j;
				k++;

				for (kk = 0; kk < offset; kk++)
					if (atomicCAS(&receive[j * offset + kk], -1, ii) == -1)
						break;
			}
		}
	}
}

template <int SUBWARP_SIZE>
__global__ void set_receive_send_subwarp(int nvtxs, int *xadj, int *adjncy, int *match, int *receive, int *send, int offset)
{
	int lane_id = threadIdx.x & (SUBWARP_SIZE - 1);
	int warp_id = threadIdx.x / SUBWARP_SIZE;
	int warp_num = blockDim.x / SUBWARP_SIZE;
	int ii = blockIdx.x * warp_num + warp_id;

	extern __shared__ int send_local[];

	for(int i = threadIdx.x;i < warp_num; i += blockDim.x)
		send_local[i] = 0;
	__syncthreads();

	if(ii < nvtxs && match[ii] == -1)
	{	
		int begin, end, flag;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		flag  = 0;

		for (int j = end - 1 - lane_id; j >= begin; j -= SUBWARP_SIZE)
		{
			int adj_vertex = adjncy[j];
			if(match[adj_vertex] == -1)
			{
				int k = atomicAdd(&send_local[warp_id], 1);
				if(k >= offset)
				{
					flag = 1;
					flag = __shfl_sync(0xffffffff, flag, lane_id, SUBWARP_SIZE);
					break;
				}
				send[ii * offset + k] = adj_vertex;
				for (int kk = 0; kk < offset; kk++)
					if (atomicCAS(&receive[adj_vertex * offset + kk], -1, ii) == -1)
						break;
			}

			if(flag)
				break;
		}
	}
}

__global__ void set_match_topk(int nvtxs, int *match, int *receive, int *send, int offset)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs && match[ii] == -1)
	{
		int i, j, k, kk;
		int begin, end, flag;
		begin = ii * offset;
		end = (ii + 1) * offset;
		flag = 0;

		for (i = begin; i < end; i++)
		{
			j = send[i];
			if (j != -1)
			{
				for (kk = begin; kk < end; kk++)
				{
					k = receive[kk];
					if (k == j)
					{
						if (atomicCAS(&match[j], -1, ii) == -1)
						{
							atomicExch(&match[ii], j);
							flag = 1;
							break;
						}
					}
				}
			}
			if (flag == 1)
				break;
		}
	}
}

__global__ void set_match_topk_serial(int nvtxs, int *match, int *receive, int *send, int offset)
{
	for (int ii = 0; ii < nvtxs; ii++)
	{
		int flag = 0;
		for (int i = 0; i < offset; i++)
		{
			int j = send[ii * offset + i];
			if (j != -1)
			{
				for (int kk = 0; kk < offset; kk++)
				{
					int k = receive[ii * offset + kk];
					if (k == j)
					{
						if (atomicCAS(&match[j], -1, ii) == -1)
						{
							atomicExch(&match[ii], j);
							flag = 1;
							break;
						}
					}
				}
			}
			if (flag == 1)
				break;
		}
	}
}

__global__ void reset_receive_send(int nvtxs, int *receive, int *send)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		receive[ii] = -1;
		send[ii] = -1;
	}
}

/*CUDA-set conflict array*/												/*cuda_cleanv*/
/*CUDA-find cgraph vertex part1-remark the match array by s*/			/*findc1*/
/*CUDA-find cgraph vertex part2-make sure the pair small label vertex*/ /*findc2*/
__global__ void resolve_conflict_1(int *match, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int t = match[ii];
		if ((t != -1 && match[t] != ii) || t == -1)
			match[ii] = ii;
		// if ((t != -1 && atomicAdd(&match[t],0) != ii) || t == -1)
		// 	atomicExch(&match[ii], ii);
	}
}

__global__ void resolve_conflict_2(int *match, int *cmap, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		if (ii == 0)
			cmap[ii] = 0;
		else
		{
			int t = match[ii];
			if (ii <= t)
				cmap[ii] = 1;
			else
				cmap[ii] = 0;
		}
	}
}

/*CUDA-find cgraph vertex part4-make sure vertex pair real rdge*/ /*findc4*/
__global__ void resolve_conflict_4(int *match, int *cmap, int *txadj, int *xadj, int *cvwgt, int *vwgt, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int u = match[ii];
		if (ii > u)
		{
			int t = cmap[u];
			cmap[ii] = t;
			cvwgt[t] = vwgt[ii] + vwgt[u];
		}
		else
		{
			int t, begin, end, length;
			t = cmap[ii];
			begin = xadj[ii];
			end = xadj[ii + 1];
			length = end - begin;
			if (u != ii)
			{
				begin = xadj[u];
				end = xadj[u + 1];
				txadj[t + 1] = length + end - begin;
			}
			else
				txadj[t + 1] = length;

			if (ii == u)
				cvwgt[t] = vwgt[ii];
		}
		if (ii == 0)
			txadj[0] = 0;
	}
}

__global__ void exam_send_receive(int nvtxs, int *receive, int *send, int offset)
{
	for (int i = 0; i < nvtxs && i < 200; i++)
	{
		printf("i=%10d receive:", i);
		for (int j = i * offset; j < (i + 1) * offset; j++)
			printf("%d ", receive[j]);
		printf("\n");
		printf("i=%10d send:  ", i);
		for (int j = i * offset; j < (i + 1) * offset; j++)
			printf("%d ", send[j]);
		printf("\n");
	}
}

__global__ void exam_match(int nvtxs, int *match)
{
	for (int i = 0; i < nvtxs && i < 200; i++)
	{
		printf("i=%10d %d\n", i, match[i]);
	}
	printf("\n");
}

__global__ void init_bin(int l, int *bin)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < l)
		bin[ii] = 0;
}

__global__ void check_length(int nvtxs, int *xadj, int *adjncy, int *length, int *bin)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int begin = xadj[ii];
		int end = xadj[ii + 1];
		int l = end - begin;
		length[ii] = l;
		
		if(l == 0)
			atomicAdd(&bin[ 0], 1);
		else if(l < 2)
			atomicAdd(&bin[ 1], 1);
		else if(l < 4)
			atomicAdd(&bin[ 2], 1);
		else if(l < 8)
			atomicAdd(&bin[ 3], 1);
		else if(l < 16)
			atomicAdd(&bin[ 4], 1);
		else if(l < 32)
			atomicAdd(&bin[ 5], 1);
		else if(l < 64)
			atomicAdd(&bin[ 6], 1);
		else if(l < 128)
			atomicAdd(&bin[ 7], 1);
		else if(l < 256)
			atomicAdd(&bin[ 8], 1);
		else if(l < 256)
			atomicAdd(&bin[ 9], 1);
		else if(l < 512)
			atomicAdd(&bin[10], 1);
		else if(l < 1024)
			atomicAdd(&bin[11], 1);
		else if(l < 2048)
			atomicAdd(&bin[12], 1);
		else if(l < 4096)
			atomicAdd(&bin[13], 1);
	}
}

__global__ void set_bin(int nvtxs, int *length, int *size, int *offset, int *idx)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int l = length[ii];
		int ptr = 0;

		if(l == 0)
		{
			ptr = atomicAdd(&size[ 0], 1);
			idx[offset[ 0] + ptr] = ii;
		}
		else if(l < 2)
		{
			ptr = atomicAdd(&size[ 1], 1);
			idx[offset[ 1] + ptr] = ii;
		}
		else if(l < 4)
		{
			ptr = atomicAdd(&size[ 2], 1);
			idx[offset[ 2] + ptr] = ii;
		}
		else if(l < 8)
		{
			ptr = atomicAdd(&size[ 3], 1);
			idx[offset[ 3] + ptr] = ii;
		}
		else if(l < 16)
		{
			ptr = atomicAdd(&size[ 4], 1);
			idx[offset[ 4] + ptr] = ii;
		}
		else if(l < 32)
		{
			ptr = atomicAdd(&size[ 5], 1);
			idx[offset[ 5] + ptr] = ii;
		}
		else if(l < 64)
		{
			ptr = atomicAdd(&size[ 6], 1);
			idx[offset[ 6] + ptr] = ii;
		}
		else if(l < 128)
		{
			ptr = atomicAdd(&size[ 7], 1);
			idx[offset[ 7] + ptr] = ii;
		}
		else if(l < 256)
		{
			ptr = atomicAdd(&size[ 8], 1);
			idx[offset[ 8] + ptr] = ii;
		}
		else if(l < 256)
		{
			ptr = atomicAdd(&size[ 9], 1);
			idx[offset[ 9] + ptr] = ii;
		}
		else if(l < 512)
		{
			ptr = atomicAdd(&size[10], 1);
			idx[offset[10] + ptr] = ii;
		}
		else if(l < 1024)
		{
			ptr = atomicAdd(&size[11], 1);
			idx[offset[11] + ptr] = ii;
		}
		else if(l < 2048)
		{
			ptr = atomicAdd(&size[12], 1);
			idx[offset[12] + ptr] = ii;
		}
		else if(l < 4096)
		{
			ptr = atomicAdd(&size[13], 1);
			idx[offset[13] + ptr] = ii;
		}
	}
}

__global__ void check_match(int nvtxs, int *match, int *length, int *bin)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int l = length[ii];
		int v = match[ii];
		if(v != -1 && ii != v && match[v] == ii)
		{
			if(l == 0)
				atomicAdd(&bin[ 0], 1);
			else if(l < 2)
				atomicAdd(&bin[ 1], 1);
			else if(l < 4)
				atomicAdd(&bin[ 2], 1);
			else if(l < 8)
				atomicAdd(&bin[ 3], 1);
			else if(l < 16)
				atomicAdd(&bin[ 4], 1);
			else if(l < 32)
				atomicAdd(&bin[ 5], 1);
			else if(l < 64)
				atomicAdd(&bin[ 6], 1);
			else if(l < 128)
				atomicAdd(&bin[ 7], 1);
			else if(l < 256)
				atomicAdd(&bin[ 8], 1);
			else if(l < 256)
				atomicAdd(&bin[ 9], 1);
			else if(l < 512)
				atomicAdd(&bin[10], 1);
			else if(l < 1024)
				atomicAdd(&bin[11], 1);
			else if(l < 2048)
				atomicAdd(&bin[12], 1);
			else if(l < 4096)
				atomicAdd(&bin[13], 1);
		}
	}
}

__global__ void print_length(int l, int *num)
{
	for(int i = 0;i < l;i++)
		printf("%10d ", num[i]);
}

__global__ void print_rate(int l, int *length_bin, int *match_bin)
{
	int ans = 0;
	for(int i = 0;i < l;i++)
	{	
		if(match_bin[i] != 0)
			printf("%10.2lf ", (double)match_bin[i] / (double)length_bin[i]);
		else 
			printf("%10d ", ans);
	}
}

__global__ void check_connection(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *match, int *length, int *bin)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int begin, end, flag, v;
		v = match[ii];
		if(v == -1)
		{
			begin = xadj[ii];
			end   = xadj[ii + 1];
			flag  = 0;
			for(int i = begin;i < end;i++)
			{
				int j = adjncy[i];
				if(match[j] == -1)
				{
					flag = 1;
					break;
				}
			}
			// flag  = 0;
			if(!flag)
			{
				int l = length[ii];
				if(l == 0)
					atomicAdd(&bin[ 0], 1);
				else if(l < 2)
					atomicAdd(&bin[ 1], 1);
				else if(l < 4)
					atomicAdd(&bin[ 2], 1);
				else if(l < 8)
					atomicAdd(&bin[ 3], 1);
				else if(l < 16)
					atomicAdd(&bin[ 4], 1);
				else if(l < 32)
					atomicAdd(&bin[ 5], 1);
				else if(l < 64)
					atomicAdd(&bin[ 6], 1);
				else if(l < 128)
					atomicAdd(&bin[ 7], 1);
				else if(l < 256)
					atomicAdd(&bin[ 8], 1);
				else if(l < 256)
					atomicAdd(&bin[ 9], 1);
				else if(l < 512)
					atomicAdd(&bin[10], 1);
				else if(l < 1024)
					atomicAdd(&bin[11], 1);
				else if(l < 2048)
					atomicAdd(&bin[12], 1);
				else if(l < 4096)
					atomicAdd(&bin[13], 1);
			}
		}
	}
}

//	leaf matches step1
__global__ void leaf_matches_step1(int nvtxs, int *xadj, int *adjncy, int *match, int *length, int *tmp_match)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && match[ii] == -1 && length[ii] == 1)
	{
		int begin = xadj[ii];
		int adj_vertex = adjncy[begin];

		atomicAdd(&tmp_match[adj_vertex], 1);
	}
}

//	leaf matches step2
__global__ void leaf_matches_step2(int nvtxs, int *xadj, int *adjncy, int *match, int *length, int *tmp_match)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && tmp_match[ii] > 1)
	{
		int begin, end, flag;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		flag = -1;

		for(int j = begin; j < end; j++)
		{
			int adj_vertex = adjncy[j];

			if(match[adj_vertex] == -1 && length[adj_vertex] == 1)
			{
				if(flag == -1)
					flag = adj_vertex;
				else 
				{
					match[flag] = adj_vertex;
					match[adj_vertex] = flag;
					flag = -1;
				}
			}
		}
	}
}

__device__ int prefixsum_subwarp(int val, int lane_id, int subwarp_size)
{
	int t = 1;
	int range = subwarp_size >> 1;
	while(t <= range)
	{
		if(lane_id >= t)
			val += __shfl_up_sync(0xffffffff, val, t, subwarp_size);
		
		t <<= 1;
	}
}
/*
adjncy: 3   7   8   9
flag:   1   0   1   1
atomicAddptr:0
vertex: 
----------------------
1:      
atomicAddptr:1
vertex: 3
2:      
atomicAddptr:2
vertex: 3   8
3:      
atomicAddptr:3
vertex: 3   8   9
----------------------
*/
template <int SUBWARP_SIZE>
__global__ void leaf_matches_step2_subwarp(int num, int *xadj, int *adjncy, int bin, int *offset, int *idx, int *match, int *length, int *tmp_match)
{
	int lane_id = threadIdx.x & (SUBWARP_SIZE - 1);
	int warp_id = threadIdx.x / SUBWARP_SIZE;
	int warp_num = blockDim.x / SUBWARP_SIZE;
	int ii = blockIdx.x * warp_num + warp_id;
	int start = offset[bin];

	extern __shared__ int cache[];
	int *id = cache + warp_id * SUBWARP_SIZE;
	int *ptr = cache + blockDim.x;

	for(int i = threadIdx.x;i < warp_num;i += blockDim.x)
		ptr[i] = 0;
	__syncthreads();

	if(ii < num)
	{	
		int vertex = idx[start + ii];

		int begin, end, flag, val, u;
		begin = xadj[vertex];
		end   = xadj[vertex + 1];
		flag  = 0;

		/*for(int j = begin + lane_id; j < end; j += SUBWARP_SIZE)
		{
			int adj_vertex = adjncy[j];

			if(match[adj_vertex] == -1 && length[adj_vertex] == 1)
			{
				flag = 1;
				id[lane_id] = vertex;
				val = atomicAdd(&ptr[lane_id], 1);

				if(val & 1)
				{
					u = id[val - 1];
					match[u] = vertex;
					match[vertex] = u;
				}
			}
		}*/
		int j = begin + lane_id;
		int adj_vertex = adjncy[j];

		if(match[adj_vertex] == -1 && length[adj_vertex] == 1)
		{
			flag = 1;
			id[lane_id] = vertex;
			val = atomicAdd(&ptr[lane_id], 1);

			if(val & 1)
			{
				u = id[val - 1];
				match[u] = vertex;
				match[vertex] = u;
			}
		}

	}
}

//	twin matches
__global__ void twin_matches_step1(int nvtxs, int *xadj, int *adjncy, int *match, int *length, int *tmp_match)
{
	int lane_id = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;
	int ii      = blockIdx.x * 4 + warp_id;

	if(ii < nvtxs && match[ii] == -1)
	{
		int begin = xadj[ii];
		int end   = xadj[ii + 1];
		long long sum = 0;
		int l     = length[ii];
		for(int j = begin + lane_id; j < end; j += 32)
			sum += adjncy[j];
		__syncwarp();

		sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

		if(lane_id == 0)
		{
			int key = sum % nvtxs;
			int hashadr = key;
			while(1)
			{
				int keyexist = tmp_match[hashadr];
                if (keyexist != -1 && length[keyexist] == l) 
				{
					atomicExch(&tmp_match[hashadr], -1);
					match[ii] = keyexist;
					match[keyexist] = ii;
					break;
				}
				else if(keyexist == -1)
				{
					if(atomicCAS(&tmp_match[hashadr], -1, ii) == -1)
					{
						break;
					}
				}
				else 
				{
					hashadr++;
					if(hashadr == nvtxs)
						hashadr = 0;
				}
			}
		}
	}
}

//	isolate matches
__global__ void isolate_matches_step1(int num, int *offset, int *idx, int *match)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < num)
	{
		int id = idx[ii];
		if(match[id] == -1 && (ii & 1) == 1)
		{
			int v = idx[ii - 1];
			match[v] = id;
			match[id] = v;
		}
	}
}

//	relative matches
__global__ void relative_matches_step1(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *match, int *length, int *tmp_match, int *mark)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && match[ii] == -1)
	{
		int begin, end, dist, min_l, l, wgt;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		dist  = -1;
		
		for(int j = begin;j < end;j++)
		{
			int adj_vertex = adjncy[j];
			
			if(dist == -1)
			{
				dist  = adj_vertex;
				min_l = length[adj_vertex];
				wgt   = adjwgt[j];
				continue;
			}

			l = length[adj_vertex];
			if(l > min_l)
			{
				dist  = adj_vertex;
				min_l = l;
				wgt   = adjwgt[j];
			}
			else if(l == min_l)
			{
				if(adjwgt[j] > wgt)
				{
					dist  = adj_vertex;
					wgt   = adjwgt[j];
				}
			}
		}

		if(dist != -1)
		{
			atomicAdd(&tmp_match[dist], 1);
			mark[ii] = 1;
		}
	}
}

__global__ void relative_matches_step2(int nvtxs, int *xadj, int *adjncy, int *match, int *tmp_match, int *mark)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && tmp_match[ii] > 1)
	{
		int begin, end, flag;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		flag  = 0;
		
		for(int j = begin;j < end;j++)
		{
			int adj_vertex = adjncy[j];
			
			if(mark[adj_vertex] == 1)
			{
				if(flag == 0)
				{
					flag = adj_vertex;
				}
				else 
				{
					match[flag] = adj_vertex;
					match[adj_vertex] = flag;
					flag = 0;
				}
			}
		}
	}
}

__global__ void check_graph(int nvtxs, int nedges, int *vwgt, int *xadj, int *adjncy, int *adjwgt)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		//	vwgt
		if(vwgt[ii] < 1)
		{
			printf("vertex %10d's vwgt is error: %10d\n", ii, vwgt[ii]);
		}
		else
		{
			//	xadj
			if(xadj[ii] < 0 || xadj[ii] > nedges ||
				xadj[ii + 1] < 0 || xadj[ii + 1] > nedges)
			{
				printf("vertex %10d's xadj is error: %10d %10d\n", ii, xadj[ii], xadj[ii + 1]);
			}
			else
			{
				int begin, end, j, k, adj_vertex, wgt_vertex, is_right;
				begin = xadj[ii];
				end   = xadj[ii + 1];
				is_right = 0;
				// adjncy
				for(j = begin;j < end;j++)
				{
					adj_vertex = adjncy[j];
					if(adj_vertex < 0 || adj_vertex >= nvtxs)
					{
						printf("vertex %10d's adjncy value is error: %10d-%10d\n", ii, j, adj_vertex);
						is_right = 0;
						break;
					}

					is_right = 0;
					for(k = xadj[adj_vertex];k < xadj[adj_vertex + 1];k++)
					{
						if(ii == adjncy[k])
						{
							is_right = 1;
							break;
						}
					}

					if(!is_right)
						printf("vertex %10d's adjncy is error: %10d-%10d\n", ii, j, adj_vertex);
				}

				//	adjwgt
				if(is_right)
				{
					is_right = 0;
					for(j = begin;j < end;j++)
					{
						adj_vertex = adjncy[j];
						wgt_vertex = adjwgt[j];
						if(wgt_vertex < 1)
						{
							printf("vertex %10d's adjwgt value is error: %10d-%10d-%10d\n", ii, j, adj_vertex, wgt_vertex);
							is_right = 0;
							break;
						}

						is_right = 0;
						for(k = xadj[adj_vertex];k < xadj[adj_vertex + 1];k++)
						{
							if(ii == adjncy[k] || wgt_vertex == adjwgt[k])
							{
								is_right = 1;
								break;
							}
						}

						if(!is_right)
							printf("vertex %10d's adjwgt is error: %10d-%10d-%10d\n", ii, j, adj_vertex, wgt_vertex);
					}
				}
			}
		}
	}
}

__global__ void print_graph(int nvtxs, int nedges, int *vwgt, int *xadj, int *adjncy, int *adjwgt)
{
	int i, j, k, begin, end;
	for(i = 0;i < nvtxs;i++)
	{
		begin = xadj[i];
		end   = xadj[i + 1];
		printf("%10d %10d %10d %10d\n", i, vwgt[i], begin, end);
		for(j = begin;j < end;j++)
			printf("%10d ", adjncy[j]);
		printf("\n");
		for(j = begin;j < end;j++)
			printf("%10d ", adjwgt[j]);
		printf("\n");
	}
}

__global__ void print_match(int nvtxs, int *match)
{
	int i;
	printf("match:");
	for(i = 0;i < nvtxs;i++)
		printf("%10d ", match[i]);
	printf("\n");
}

/*Get gpu graph matching params by hem*/
hunyuangraph_graph_t *hunyuangraph_gpu_match(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int level)
{
	int nvtxs = graph->nvtxs;
	int nedges = graph->nedges;
	int cnvtxs = 0;

	// tesst
	int success_num[5];
	double success_rate[5];

	// cudaDeviceSynchronize();
	// print_graph<<<1, 1>>>(nvtxs, nedges, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
	// cudaDeviceSynchronize();

	// cudaDeviceSynchronize();
	// check_graph<<<(nvtxs + 127) / 128, 128>>>(nvtxs, nedges, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
	// cudaDeviceSynchronize();

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	init_gpu_match<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	init_gpu_match_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	int *length_vertex, *length_bin, *bin_offset, *bin_size, *bin_idx, *match_bin, *match_num;
	double sum;
	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc,NULL);
	if(GPU_Memory_Pool)
	{
		length_vertex = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_gpu_match: length_vertex");
		length_bin = (int *)rmalloc_with_check(sizeof(int) * 14, "hunyuangraph_gpu_match: length_bin");
		bin_offset = (int *)rmalloc_with_check(sizeof(int) * 15, "hunyuangraph_gpu_match: bin_offset");
		bin_size   = (int *)rmalloc_with_check(sizeof(int) * 14, "hunyuangraph_gpu_match: bin_size");
		bin_idx    = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_gpu_match: bin_idx");
		match_bin = (int *)rmalloc_with_check(sizeof(int) * 14, "hunyuangraph_gpu_match: match_bin");
	}
	else 
	{
		cudaMalloc((void**)&length_vertex, sizeof(int) * nvtxs);
		cudaMalloc((void**)&length_bin, sizeof(int) * 14);
		cudaMalloc((void**)&bin_offset, sizeof(int) * 15);
		cudaMalloc((void**)&bin_size, sizeof(int) * 14);
		cudaMalloc((void**)&bin_idx, sizeof(int) * nvtxs);
		cudaMalloc((void**)&match_bin, sizeof(int) * 14);
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc,NULL);
	coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
	
	match_num = (int *)malloc(sizeof(int) * 14);

	cudaDeviceSynchronize();
	init_bin<<<1, 14>>>(14, length_bin);
	init_bin<<<1, 14>>>(14, match_bin);
	init_bin<<<1, 14>>>(15, bin_offset);
	init_bin<<<1, 14>>>(14, bin_size);
	// cudaDeviceSynchronize();
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	check_length<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, length_vertex, length_bin);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	check_length_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	cudaMemcpy(&bin_offset[1], length_bin, sizeof(int) * 14, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	if(GPU_Memory_Pool)
    {
        prefixsum(bin_offset, bin_offset, 15, prefixsum_blocksize, 1);	//0:lmalloc,1:rmalloc
    }
    else
    {
        // thrust::exclusive_scan(thrust::device, bin_offset, bin_offset + 15, bin_offset);
		thrust::inclusive_scan(thrust::device, bin_offset, bin_offset + 15, bin_offset);
    }
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	set_bin<<<(nvtxs + 127) / 128, 128>>>(nvtxs, length_vertex, bin_size, bin_offset, bin_idx);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	set_bin_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	// cudaDeviceSynchronize();
	// print_length<<<1, 1>>>(14, length_bin);
	// cudaDeviceSynchronize();
	// printf("\n");
	// cudaDeviceSynchronize();
	// print_length<<<1, 1>>>(15, bin_offset);
	// cudaDeviceSynchronize();
	// printf("\n");
	// cudaDeviceSynchronize();
	// print_length<<<1, 1>>>(14, bin_size);
	// cudaDeviceSynchronize();
	// printf("\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);

	if (level == 0)
	{
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		// SC24_version
		for (int i = 0; i < 1; i++)
		{
			cuda_hem_229_3<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, graph->cuda_xadj, graph->cuda_vwgt, graph->cuda_adjwgt, graph->cuda_adjncy,
														 hunyuangraph_admin->maxvwgt);

			reset_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match);

			cuda_hem<<<1024, 1>>>(nvtxs, graph->cuda_match, graph->cuda_xadj, graph->cuda_vwgt, graph->cuda_adjwgt, graph->cuda_adjncy,
								  hunyuangraph_admin->maxvwgt);
		}

		// SC25 random version
		// random_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match);

		// random_match_group<<<1024, 1>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match);

		// reset_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match);

		// cudaDeviceSynchronize();
		// exam_match<<<1,1>>>(nvtxs, graph->cuda_match);
		// cudaDeviceSynchronize();

		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		random_match_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
	}

	// SC25 topk version
	else
	{
		cudaDeviceSynchronize();
	    gettimeofday(&begin_gpu_topkfour_match,NULL);
		// printf("topk begin\n");

		int *receive, *send;
		cudaDeviceSynchronize();
	    gettimeofday(&begin_malloc,NULL);
		if(GPU_Memory_Pool)
		{
			receive = (int *)rmalloc_with_check(sizeof(int) * nvtxs * 4, "hunyuangraph_gpu_match: receive");
			send = (int *)rmalloc_with_check(sizeof(int) * nvtxs * 4, "hunyuangraph_gpu_match: send");
		}
		else
		{
			cudaMalloc((void**)&receive, sizeof(int) * nvtxs * 4);
			cudaMalloc((void**)&send, sizeof(int) * nvtxs * 4);
		}
		cudaDeviceSynchronize();
		gettimeofday(&end_malloc,NULL);
		match_time = (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
		coarsen_malloc += match_time;
		match_malloc_time += match_time;

		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		init_gpu_receive_send<<<(nvtxs * 4 + 127) / 128, 128>>>(receive, nvtxs * 4);
		init_gpu_receive_send<<<(nvtxs * 4 + 127) / 128, 128>>>(send, nvtxs * 4);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		init_gpu_receive_send_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
		// printf("segmengtsort begin\n");
		// topk sort
		if(GPU_Memory_Pool)
		{
			int *bb_counter, *bb_id;
			int *bb_keysB_d, *bb_valsB_d;

			cudaDeviceSynchronize();
			gettimeofday(&begin_malloc,NULL);
			bb_keysB_d = (int *)rmalloc_with_check(sizeof(int) * nedges, "bb_keysB_d");
			bb_valsB_d = (int *)rmalloc_with_check(sizeof(int) * nedges, "bb_valsB_d");
			bb_id = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "bb_id");
			bb_counter = (int *)rmalloc_with_check(sizeof(int) * 13, "bb_counter");
			cudaDeviceSynchronize();
			gettimeofday(&end_malloc,NULL);
			match_time = (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
			coarsen_malloc += match_time;
			match_malloc_time += match_time;

			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_match_kernel, NULL);
			hunyuangraph_segmengtsort(graph->cuda_adjwgt, graph->cuda_adjncy, nedges, graph->cuda_xadj, nvtxs, bb_counter, bb_id, bb_keysB_d, bb_valsB_d);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_match_kernel, NULL);
			wgt_segmentsort_gpu_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
			// printf("hunyuangraph_gpu_match hunyuangraph_segmengtsort end\n");

			cudaDeviceSynchronize();
	        gettimeofday(&begin_free,NULL);
			rfree_with_check((void *)bb_counter, sizeof(int) * 13, "bb_counter");	// bb_counter
			rfree_with_check((void *)bb_id, sizeof(int) * nvtxs, "bb_id");			// bb_id
			cudaDeviceSynchronize();
			gettimeofday(&end_free,NULL);
			match_time = (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
			coarsen_free += match_time;
			match_free_time += match_time;

			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_match_kernel, NULL);
			cudaMemcpy(graph->cuda_adjncy, bb_valsB_d, sizeof(int) * nedges, cudaMemcpyDeviceToDevice);
			cudaMemcpy(graph->cuda_adjwgt, bb_keysB_d, sizeof(int) * nedges, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_match_kernel, NULL);
			match_time = (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
			match_memcpy_time += match_time;
			segmentsort_memcpy_time += match_time;
			
			cudaDeviceSynchronize();
	        gettimeofday(&begin_free,NULL);
			rfree_with_check((void *)bb_valsB_d, sizeof(int) * nedges, "bb_valsB_d");	// bb_valsB_d
			rfree_with_check((void *)bb_keysB_d, sizeof(int) * nedges, "bb_keysB_d");	// bb_keysB_d
			cudaDeviceSynchronize();
			gettimeofday(&end_free,NULL);
			match_time = (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
			coarsen_free += match_time;
			match_free_time += match_time;
		}
		else
		{
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_match_kernel, NULL);
			bb_segsort(graph->cuda_adjwgt, graph->cuda_adjncy, nedges, graph->cuda_xadj, nvtxs);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_match_kernel, NULL);
			wgt_segmentsort_gpu_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
		}

		// cudaDeviceSynchronize();
		// exam_csr<<<1, 1>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
		// cudaDeviceSynchronize();

		// cudaDeviceSynchronize();
		// print_graph<<<1, 1>>>(10, nedges, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
		// cudaDeviceSynchronize();

		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_topkfour_match,NULL);
		top1_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;

		// 4 iteration
		for (int iter = 0; iter < 4; iter++)
		{
			cudaDeviceSynchronize();
		    gettimeofday(&begin_gpu_topkfour_match,NULL);
			int offset = iter + 1;

			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_match_kernel, NULL);
			// set_receive_send<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_match, receive, send, offset);
			switch (offset)
			{
			case 1:
				// set_receive_send_subwarp<1><<<(nvtxs + 127) / 128, 128, sizeof(int) * 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match, receive, send, offset);
				set_receive_send<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_match, receive, send, offset);
				break;
			case 2:
				set_receive_send_subwarp<2><<<(nvtxs * 2 + 127) / 128, 128, sizeof(int) * 64>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match, receive, send, offset);
				break;
			case 3:
				set_receive_send_subwarp<4><<<(nvtxs * 4 + 127) / 128, 128, sizeof(int) * 32>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match, receive, send, offset);
				break;
			case 4:
				set_receive_send_subwarp<4><<<(nvtxs * 4 + 127) / 128, 128, sizeof(int) * 32>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match, receive, send, offset);
				break;
			default:
				break;
			}
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_match_kernel, NULL);
			set_receive_send_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
			// printf("hunyuangraph_gpu_match set_receiver_send iter=%d end\n",iter);

			// cudaDeviceSynchronize();
			// exam_send_receive<<<1, 1>>>(nvtxs, receive, send, offset);
			// cudaDeviceSynchronize();
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_match_kernel, NULL);
			set_match_topk<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, receive, send, offset);
			// set_match_topk_serial<<<1, 1>>>(nvtxs, graph->cuda_match, receive, send, offset);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_match_kernel, NULL);
			set_match_topk_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
			// printf("hunyuangraph_gpu_match set_match_topk iter=%d end\n",iter);

			// cudaDeviceSynchronize();
			// exam_match<<<1,1>>>(nvtxs, graph->cuda_match);
			// cudaDeviceSynchronize();

			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_match_kernel, NULL);
			reset_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match);
			reset_receive_send<<<(nvtxs * offset + 127) / 128, 128>>>(nvtxs * offset, receive, send);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_match_kernel, NULL);
			reset_match_array_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
			// printf("hunyuangraph_gpu_match reset_match iter=%d end\n",iter);

			// int *host_match = (int *)malloc(sizeof(int) * nvtxs);
			// cudaMemcpy(host_match, graph->cuda_match, nvtxs * sizeof(int), cudaMemcpyDeviceToHost);
			// success_num[iter] = 0;
			// for (int i = 0; i < nvtxs; i++)
			// {
			// 	if (host_match[i] != -1 && host_match[i] != i && host_match[host_match[i]] == i)
			// 	{
			// 		success_num[iter]++;
			// 	}
			// }
			// free(host_match);
			// success_rate[iter] = (double)success_num[iter] / (double)nvtxs * 100;
			// printf("iter=%2d success_rate=%10.2lf%% success_num=%10d the iter add %10.2lf%%\n", iter, success_rate[iter], success_num[iter], success_rate[iter] - success_rate[iter - 1]);
			
			// cudaDeviceSynchronize();
			// init_bin<<<1, 14>>>(14, match_bin);
			// cudaDeviceSynchronize();
			// check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
			// cudaDeviceSynchronize();
			// print_length<<<1, 1>>>(14, match_bin);
			// cudaDeviceSynchronize();
			// printf("\n");
			// cudaDeviceSynchronize();
			// print_rate<<<1, 1>>>(14, length_bin, match_bin);
			// cudaDeviceSynchronize();
			// printf("\n");

			// cudaDeviceSynchronize();
			// print_match<<<1, 1>>>(10, graph->cuda_match);
			// cudaDeviceSynchronize();

			switch (offset)
			{
			case 1:
				cudaDeviceSynchronize();
				gettimeofday(&end_gpu_topkfour_match,NULL);
				top1_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;
				break;
			case 2:
				cudaDeviceSynchronize();
				gettimeofday(&end_gpu_topkfour_match,NULL);
				top2_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;
				break;
			case 3:
				cudaDeviceSynchronize();
				gettimeofday(&end_gpu_topkfour_match,NULL);
				top3_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;
				break;
			case 4:
				cudaDeviceSynchronize();
				gettimeofday(&end_gpu_topkfour_match,NULL);
				top4_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;
				break;
			default:
				break;
			}
		}

		cudaDeviceSynchronize();
	    gettimeofday(&begin_gpu_topkfour_match,NULL);

		cudaDeviceSynchronize();
	    gettimeofday(&begin_free,NULL);
		if(GPU_Memory_Pool)
		{
			rfree_with_check((void *)send, sizeof(int) * nvtxs * 4, "hunyuangraph_gpu_match: send");		// send
			rfree_with_check((void *)receive, sizeof(int) * nvtxs * 4, "hunyuangraph_gpu_match: receive");	// receive
		}
		else
		{
			cudaFree(send);
			cudaFree(receive);
		}
		cudaDeviceSynchronize();
		gettimeofday(&end_free,NULL);
		match_time = (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
		coarsen_free += match_time;
		match_free_time += match_time;

		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_topkfour_match,NULL);
		top1_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;
	}

	cudaDeviceSynchronize();
	init_bin<<<1, 14>>>(14, match_bin);
	cudaDeviceSynchronize();
	check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
	cudaDeviceSynchronize();
	cudaMemcpy(match_num, match_bin, sizeof(int) * 14, cudaMemcpyDeviceToHost);
	sum = 0;
	for(int i = 0;i < 14;i++)
		sum += match_num[i];
	// print_length<<<1, 1>>>(14, match_bin);
	// cudaDeviceSynchronize();
	// printf("\n");
	// cudaDeviceSynchronize();
	// print_rate<<<1, 1>>>(14, length_bin, match_bin);
	// cudaDeviceSynchronize();
	// printf("\n");

	// cudaDeviceSynchronize();
	// init_bin<<<1, 14>>>(14, match_bin);
	// cudaDeviceSynchronize();
	// check_connection<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_match, length_vertex, match_bin);
	// cudaDeviceSynchronize();
	// print_length<<<1, 1>>>(14, match_bin);
	// cudaDeviceSynchronize();
	// printf("\n");

	// printf("sum=%10.0lf\n", sum);
	//	leaf matches
	cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_topkfour_match,NULL);
	if(sum / (double)nvtxs < 0.5)
	{
		// printf("sum=%10.0lf leaf matches begin\n", sum);
		int *tmp_match;

		cudaDeviceSynchronize();
		gettimeofday(&begin_malloc,NULL);
		if(GPU_Memory_Pool)
			tmp_match = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "tmp_match");
		else 
			cudaMalloc((void**)&tmp_match, sizeof(int) * nvtxs);
		cudaDeviceSynchronize();
		gettimeofday(&end_malloc,NULL);
		match_time = (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
		coarsen_malloc += match_time;
		match_malloc_time += match_time;

		cudaDeviceSynchronize();
		init_bin<<<(nvtxs + 127) / 128, 128>>>(nvtxs, tmp_match);
		// init_gpu_match<<<(nvtxs + 127) / 128, 128>>>(tmp_match, nvtxs);
		// cudaDeviceSynchronize();

		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		leaf_matches_step1<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match, length_vertex, tmp_match);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		leaf_matches_step1_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;

		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		leaf_matches_step2<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match, length_vertex, tmp_match);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		leaf_matches_step2_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;

		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		reset_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		reset_match_array_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
		
		cudaDeviceSynchronize();
	    gettimeofday(&begin_free,NULL);
		if(GPU_Memory_Pool)
			rfree_with_check((void *)tmp_match, sizeof(int) * nvtxs, "tmp_match");	//	tmp_match
		else
			cudaFree(tmp_match);
		cudaDeviceSynchronize();
		gettimeofday(&end_free,NULL);
		match_time = (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
		coarsen_free += match_time;
		match_free_time += match_time;

		// cudaDeviceSynchronize();
		// init_bin<<<1, 14>>>(14, match_bin);
		// cudaDeviceSynchronize();
		// check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
		// cudaDeviceSynchronize();
		// print_length<<<1, 1>>>(14, match_bin);
		// cudaDeviceSynchronize();
		// printf("\n");
		// cudaDeviceSynchronize();
		// print_rate<<<1, 1>>>(14, length_bin, match_bin);
		// cudaDeviceSynchronize();
		// printf("\n");
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_topkfour_match,NULL);
	leaf_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	init_bin<<<1, 14>>>(14, match_bin);
	cudaDeviceSynchronize();
	check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
	cudaDeviceSynchronize();
	cudaMemcpy(match_num, match_bin, sizeof(int) * 14, cudaMemcpyDeviceToHost);
	sum = 0;
	for(int i = 0;i < 14;i++)
		sum += match_num[i];
	
	// printf("sum=%10.0lf\n", sum);
	//	isolate matches
	cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_topkfour_match,NULL);
	if(sum / (double)nvtxs < 0.5)
	{
		// printf("sum=%10.0lf isolate matches begin\n", sum);
		int tmp_num;
		cudaMemcpy(&tmp_num, &bin_offset[1], sizeof(int), cudaMemcpyDeviceToHost);
		// printf("sum=%10.0lf isolate matches begin: have %10d isolate vertices\n", sum, tmp_num);

		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		isolate_matches_step1<<<(tmp_num + 127) / 128, 128>>>(tmp_num, bin_offset, bin_idx, graph->cuda_match);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		isolate_matches_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;

		// cudaDeviceSynchronize();
		// init_bin<<<1, 14>>>(14, match_bin);
		// cudaDeviceSynchronize();
		// check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
		// cudaDeviceSynchronize();
		// print_length<<<1, 1>>>(14, match_bin);
		// cudaDeviceSynchronize();
		// printf("\n");
		// cudaDeviceSynchronize();
		// print_rate<<<1, 1>>>(14, length_bin, match_bin);
		// cudaDeviceSynchronize();
		// printf("\n");
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_topkfour_match,NULL);
	isolate_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	init_bin<<<1, 14>>>(14, match_bin);
	cudaDeviceSynchronize();
	check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
	cudaDeviceSynchronize();
	cudaMemcpy(match_num, match_bin, sizeof(int) * 14, cudaMemcpyDeviceToHost);
	sum = 0;
	for(int i = 0;i < 14;i++)
		sum += match_num[i];

	// printf("sum=%10.0lf\n", sum);
	//	twin matches
	cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_topkfour_match,NULL);
	if(sum / (double)nvtxs < 0.5)
	{
		// printf("sum=%10.0lf twin matches begin\n", sum);
		int *tmp_match;

		cudaDeviceSynchronize();
		gettimeofday(&begin_malloc,NULL);
		if(GPU_Memory_Pool)
			tmp_match = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "tmp_match");
		else 
			cudaMalloc((void**)&tmp_match, sizeof(int) * nvtxs);
		cudaDeviceSynchronize();
		gettimeofday(&end_malloc,NULL);
		match_time = (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
		coarsen_malloc += match_time;
		match_malloc_time += match_time;

		cudaDeviceSynchronize();
		init_gpu_match<<<(nvtxs + 127) / 128, 128>>>(tmp_match, nvtxs);
		// init_bin<<<(nvtxs + 127) / 128, 128>>>(nvtxs, tmp_match);
		cudaDeviceSynchronize();

		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		twin_matches_step1<<<(nvtxs + 3) / 4, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match, length_vertex, tmp_match);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		twin_matches_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;

		cudaDeviceSynchronize();
	    gettimeofday(&begin_free,NULL);
		if(GPU_Memory_Pool)
			rfree_with_check((void *)tmp_match, sizeof(int) * nvtxs, "tmp_match");	//	tmp_match
		else 
			cudaFree(tmp_match);
		cudaDeviceSynchronize();
		gettimeofday(&end_free,NULL);
		match_time = (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
		coarsen_free += match_time;
		match_free_time += match_time;

		// cudaDeviceSynchronize();
		// init_bin<<<1, 14>>>(14, match_bin);
		// cudaDeviceSynchronize();
		// check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
		// cudaDeviceSynchronize();
		// print_length<<<1, 1>>>(14, match_bin);
		// cudaDeviceSynchronize();
		// printf("\n");
		// cudaDeviceSynchronize();
		// print_rate<<<1, 1>>>(14, length_bin, match_bin);
		// cudaDeviceSynchronize();
		// printf("\n");
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_topkfour_match,NULL);
	twin_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;

	cudaMemcpy(match_num, match_bin, sizeof(int) * 14, cudaMemcpyDeviceToHost);
	sum = 0;
	for(int i = 0;i < 14;i++)
		sum += match_num[i];
	
	// printf("sum=%10.0lf\n", sum);
	//	relative matches
	cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_topkfour_match,NULL);
	if(sum / (double)nvtxs < 0.5)
	{
		// printf("sum=%10.0lf relative matches begin\n", sum);
		int *tmp_match, *tmp_mark;

		cudaDeviceSynchronize();
		gettimeofday(&begin_malloc,NULL);
		if(GPU_Memory_Pool)
		{
			tmp_match = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "tmp_match");
			tmp_mark = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "tmp_mark");
		}
		else 
		{
			cudaMalloc((void**)&tmp_match, sizeof(int) * nvtxs);
			cudaMalloc((void**)&tmp_mark, sizeof(int) * nvtxs);
		}
		cudaDeviceSynchronize();
		gettimeofday(&end_malloc,NULL);
		match_time = (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
		coarsen_malloc += match_time;
		match_malloc_time += match_time;

		cudaDeviceSynchronize();
		init_bin<<<(nvtxs + 127) / 128, 128>>>(nvtxs, tmp_match);
		init_bin<<<(nvtxs + 127) / 128, 128>>>(nvtxs, tmp_mark);
		cudaDeviceSynchronize();

		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		relative_matches_step1<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_match, length_vertex, tmp_match, tmp_mark);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		relative_matches_step1_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
		
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		relative_matches_step2<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_match, tmp_match, tmp_mark);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		relative_matches_step2_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;
		
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_match_kernel, NULL);
		reset_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match);
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_match_kernel, NULL);
		reset_match_array_time += (end_gpu_match_kernel.tv_sec - begin_gpu_match_kernel.tv_sec) * 1000 + (end_gpu_match_kernel.tv_usec - begin_gpu_match_kernel.tv_usec) / 1000.0;

		cudaDeviceSynchronize();
	    gettimeofday(&begin_free,NULL);
		if(GPU_Memory_Pool)
		{
			rfree_with_check((void *)tmp_mark, sizeof(int) * nvtxs, "tmp_mark");	//	tmp_mark
			rfree_with_check((void *)tmp_match, sizeof(int) * nvtxs, "tmp_match");	//	tmp_match
		}
		else 
		{
			cudaFree(tmp_mark);
			cudaFree(tmp_match);
		}
		cudaDeviceSynchronize();
		gettimeofday(&end_free,NULL);
		match_time = (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
		coarsen_free += match_time;
		match_free_time += match_time;

		// cudaDeviceSynchronize();
		// init_bin<<<1, 14>>>(14, match_bin);
		// cudaDeviceSynchronize();
		// check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
		// cudaDeviceSynchronize();
		// print_length<<<1, 1>>>(14, match_bin);
		// cudaDeviceSynchronize();
		// printf("\n");
		// cudaDeviceSynchronize();
		// print_rate<<<1, 1>>>(14, length_bin, match_bin);
		// cudaDeviceSynchronize();
		// printf("\n");
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_topkfour_match,NULL);
	relative_time += (end_gpu_topkfour_match.tv_sec - begin_gpu_topkfour_match.tv_sec) * 1000 + (end_gpu_topkfour_match.tv_usec - begin_gpu_topkfour_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	init_bin<<<1, 14>>>(14, match_bin);
	cudaDeviceSynchronize();
	check_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, length_vertex, match_bin);
	cudaDeviceSynchronize();
	cudaMemcpy(match_num, match_bin, sizeof(int) * 14, cudaMemcpyDeviceToHost);
	sum = 0;
	for(int i = 0;i < 14;i++)
		sum += match_num[i];
	
	// printf("sum=%10.0lf\n", sum);

	// cudaDeviceSynchronize();
	// init_bin<<<1, 14>>>(14, match_bin);
	// cudaDeviceSynchronize();
	// check_connection<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_match, length_vertex, match_bin);
	// cudaDeviceSynchronize();
	// print_length<<<1, 1>>>(14, match_bin);
	// cudaDeviceSynchronize();
	// printf("\n");
	// print_rate<<<1, 1>>>(14, length_bin, match_bin);
	// cudaDeviceSynchronize();
	// printf("\n");
	// printf("--------------------------------------------------------------------------------------------------------------------------------------------------\n");

	cudaDeviceSynchronize();
    gettimeofday(&begin_free,NULL);
	if(GPU_Memory_Pool)
	{
		rfree_with_check((void *)match_bin, sizeof(int) * 14, "hunyuangraph_gpu_match: match_bin");				//	match_bin
		rfree_with_check((void *)bin_idx, sizeof(int) * nvtxs, "hunyuangraph_gpu_match: bin_idx");				//	bin_idx
		rfree_with_check((void *)bin_size, sizeof(int) * 14, "hunyuangraph_gpu_match: bin_size");				//	bin_size
		rfree_with_check((void *)bin_offset, sizeof(int) * 15, "hunyuangraph_gpu_match: bin_offset");			//	bin_offset
		rfree_with_check((void *)length_bin, sizeof(int) * 14, "hunyuangraph_gpu_match: length_bin");			//	length_bin
		rfree_with_check((void *)length_vertex, sizeof(int) * nvtxs, "hunyuangraph_gpu_match: length_vertex");	//	length_vertex
	}
	else
	{
		cudaFree(match_bin);
		cudaFree(bin_idx);
		cudaFree(bin_size);
		cudaFree(bin_offset);
		cudaFree(length_bin);
		cudaFree(length_vertex);
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_free,NULL);
	match_time = (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
	coarsen_free += match_time;
	match_free_time += match_time;

	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	hem_gpu_match_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	resolve_conflict_1<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	resolve_conflict_1_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	// cudaDeviceSynchronize();
	// print_match<<<1, 1>>>(10, graph->cuda_match);
	// cudaDeviceSynchronize();

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	resolve_conflict_2<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, graph->cuda_cmap, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	resolve_conflict_2_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	// cudaDeviceSynchronize();
	// print_match<<<1, 1>>>(10, graph->cuda_cmap);
	// cudaDeviceSynchronize();

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	if(GPU_Memory_Pool)
    {
        prefixsum(graph->cuda_cmap, graph->cuda_cmap, nvtxs, prefixsum_blocksize, 0); // 0:lmalloc,1:rmalloc
    }
    else
    {
        thrust::inclusive_scan(thrust::device, graph->cuda_cmap, graph->cuda_cmap + nvtxs, graph->cuda_cmap);
    }
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	inclusive_scan_time1 += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	// cudaDeviceSynchronize();
	// print_match<<<1, 1>>>(10, graph->cuda_cmap);
	// cudaDeviceSynchronize();

	cudaMemcpy(&cnvtxs, &graph->cuda_cmap[nvtxs - 1], sizeof(int), cudaMemcpyDeviceToHost);
	cnvtxs++;

	hunyuangraph_graph_t *cgraph = hunyuangraph_set_gpu_cgraph(graph, cnvtxs);
	cgraph->nvtxs = cnvtxs;

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc, NULL);
	if(GPU_Memory_Pool)
	{
		graph->txadj = (int *)rmalloc_with_check(sizeof(int) * (cnvtxs + 1), "txadj");
		cgraph->cuda_vwgt = (int *)lmalloc_with_check(sizeof(int) * cnvtxs, "vwgt");
	}
	else 
	{
		cudaMalloc((void**)&graph->txadj, sizeof(int) * (cnvtxs + 1));
		cudaMalloc((void**)&cgraph->cuda_vwgt, sizeof(int) * cnvtxs);
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc, NULL);
	coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	resolve_conflict_4<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, graph->cuda_cmap, graph->txadj, graph->cuda_xadj,
													 cgraph->cuda_vwgt, graph->cuda_vwgt, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	resolve_conflict_4_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	// cudaDeviceSynchronize();
	// print_match<<<1, 1>>>(10, graph->cuda_cmap);
	// cudaDeviceSynchronize();

	return cgraph;
}

#endif