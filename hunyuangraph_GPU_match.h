#ifndef _H_GPU_MATCH
#define _H_GPU_MATCH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_GPU_prefixsum.h"
#include "hunyuangraph_bb_segsort.h"

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

__global__ void set_receive_send(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *match, int *receive, int *send, int offset)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs && match[ii] == -1)
	{
		int i, j, k, kk;
		int begin, end;
		begin = xadj[ii];
		end = xadj[ii + 1];

		for (i = begin, k = 0; i < end && k < offset; i++)
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
	for (int i = 0; i < nvtxs; i++)
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
	for (int i = 0; i < nvtxs; i++)
	{
		printf("i=%10d %d\n", i, match[i]);
	}
	printf("\n");
}

/*Get gpu graph matching params by hem*/
hunyuangraph_graph_t *hunyuangraph_gpu_match(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nedges = graph->nedges;
	int cnvtxs = 0;

	// tesst
	int success_num;
	double success_rate;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	init_gpu_match<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	init_gpu_match_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);

	// SC24_version
	for (int i = 0; i < 1; i++)
	{
		cuda_hem_229_3<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match, graph->cuda_xadj, graph->cuda_vwgt, graph->cuda_adjwgt, graph->cuda_adjncy,
													 hunyuangraph_admin->maxvwgt);
		cudaDeviceSynchronize();

		reset_match<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_match);
		cudaDeviceSynchronize();

		cuda_hem<<<1024, 1>>>(nvtxs, graph->cuda_match, graph->cuda_xadj, graph->cuda_vwgt, graph->cuda_adjwgt, graph->cuda_adjncy,
							  hunyuangraph_admin->maxvwgt);
		cudaDeviceSynchronize();
	}

	// SC25 topk version
	/*int *receive, *send;
	receive = (int *)rmalloc_with_check(sizeof(int) * nvtxs * 5, "hunyuangraph_gpu_match: receive");
	send    = (int *)rmalloc_with_check(sizeof(int) * nvtxs * 5, "hunyuangraph_gpu_match: send");

	init_gpu_match<<<(nvtxs * 5 + 127) / 128,128>>>(receive, nvtxs * 5);
	init_gpu_match<<<(nvtxs * 5 + 127) / 128,128>>>(send, nvtxs * 5);
	cudaDeviceSynchronize();

	// topk sort
	int *bb_counter, *bb_id;
	int *bb_keysB_d, *bb_valsB_d;
	bb_keysB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"bb_keysB_d");
	bb_valsB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"bb_valsB_d");
	bb_id      = (int *)rmalloc_with_check(sizeof(int) * nvtxs,"bb_id");
	bb_counter = (int *)rmalloc_with_check(sizeof(int) * 13,"bb_counter");

	hunyuangraph_segmengtsort(graph->cuda_adjwgt, graph->cuda_adjncy, nedges, graph->cuda_xadj, nvtxs, bb_counter, bb_id, bb_keysB_d, bb_valsB_d);
	cudaDeviceSynchronize();
	// printf("hunyuangraph_gpu_match hunyuangraph_segmengtsort end\n");

	rfree_with_check(sizeof(int) * 13,"bb_counter");			//bb_counter
	rfree_with_check(sizeof(int) * nvtxs,"bb_id");				//bb_id
	cudaMemcpy(graph->cuda_adjncy, bb_valsB_d, sizeof(int) * nedges, cudaMemcpyDeviceToDevice);
	cudaMemcpy(graph->cuda_adjwgt, bb_keysB_d, sizeof(int) * nedges, cudaMemcpyDeviceToDevice);
	rfree_with_check(sizeof(int) * nedges,"bb_valsB_d");		//bb_valsB_d
	rfree_with_check(sizeof(int) * nedges,"bb_keysB_d");		//bb_keysB_d

	// exam_csr<<<1,1>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
	// cudaDeviceSynchronize();

	// 5 iteration
	for(int iter = 0;iter < 5;iter++)
	{
		int offset = iter + 1;

		set_receive_send<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_match, receive, send, offset);
		cudaDeviceSynchronize();
		// printf("hunyuangraph_gpu_match set_receiver_send iter=%d end\n",iter);

		// cudaDeviceSynchronize();
		// exam_send_receive<<<1,1>>>(nvtxs, receive, send, offset);
		// cudaDeviceSynchronize();

		set_match_topk<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->cuda_match, receive, send, offset);
		cudaDeviceSynchronize();
		// printf("hunyuangraph_gpu_match set_match_topk iter=%d end\n",iter);

		// cudaDeviceSynchronize();
		// exam_match<<<1,1>>>(nvtxs, graph->cuda_match);
		// cudaDeviceSynchronize();

		reset_match<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->cuda_match);
		reset_receive_send<<<(nvtxs * offset + 127) / 128,128>>>(nvtxs * offset, receive, send);
		cudaDeviceSynchronize();
		// printf("hunyuangraph_gpu_match reset_match iter=%d end\n",iter);

		/*int *host_match = (int *)malloc(sizeof(int) * nvtxs);
		cudaMemcpy(host_match, graph->cuda_match, nvtxs * sizeof(int), cudaMemcpyDeviceToHost);
		success_num = 0;
		for(int i = 0;i < nvtxs;i++)
		{
			if(host_match[i] != -1 && host_match[i] != i && host_match[host_match[i]] == i)
			{
				success_num++;
			}
		}
		success_rate = (double)success_num / (double)nvtxs * 100;
		printf("iter=%d success_rate=%10.2lf%% success_num=%d\n",iter, success_rate, success_num);*
	}

	rfree_with_check(sizeof(int) * nvtxs * 5, "hunyuangraph_gpu_match: send");			// send
	rfree_with_check(sizeof(int) * nvtxs * 5, "hunyuangraph_gpu_match: receive");		// receive*/

	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	hem_gpu_match_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	resolve_conflict_1<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	resolve_conflict_1_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	resolve_conflict_2<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, graph->cuda_cmap, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	resolve_conflict_2_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match, NULL);
	// thrust::inclusive_scan(thrust::device, graph->cuda_cmap, graph->cuda_cmap + nvtxs, graph->cuda_cmap);
	prefixsum(graph->cuda_cmap, graph->cuda_cmap, nvtxs, prefixsum_blocksize, 0); // 0:lmalloc,1:rmalloc
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match, NULL);
	inclusive_scan_time1 += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaMemcpy(&cnvtxs, &graph->cuda_cmap[nvtxs - 1], sizeof(int), cudaMemcpyDeviceToHost);
	cnvtxs++;

	hunyuangraph_graph_t *cgraph = hunyuangraph_set_gpu_cgraph(graph, cnvtxs);
	cgraph->nvtxs = cnvtxs;

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc, NULL);
	// cudaMalloc((void**)&graph->txadj, (cnvtxs + 1) * sizeof(int));
	graph->txadj = (int *)rmalloc_with_check(sizeof(int) * (cnvtxs + 1), "txadj");
	// cudaMalloc((void**)&cgraph->cuda_vwgt, cnvtxs * sizeof(int));
	cgraph->cuda_vwgt = (int *)lmalloc_with_check(sizeof(int) * cnvtxs, "vwgt");
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

	return cgraph;
}

#endif