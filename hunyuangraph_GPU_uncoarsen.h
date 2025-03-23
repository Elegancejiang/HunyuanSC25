#ifndef _H_GPU_UNCOARSEN
#define _H_GPU_UNCOARSEN

#include "hunyuangraph_struct.h"
#include "hunyuangraph_GPU_memory.h"
#include "hunyuangraph_GPU_krefine.h"

/*CUDA-init pwgts array*/
__global__ void initpwgts(int *cuda_pwgts, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nparts)
		cuda_pwgts[ii] = 0;
}

/*Compute sum of pwgts*/
__global__ void calculateSum(int nvtxs, int nparts, int *pwgts, int *where, int *vwgt)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int cache_d[];
	for (int i = threadIdx.x; i < nparts; i += 128)
		cache_d[i] = 0;
	__syncthreads();

	int t;
	if (ii < nvtxs)
	{
		t = where[ii];
		atomicAdd(&cache_d[t], vwgt[ii]);
	}
	__syncthreads();

	int val;
	for (int i = threadIdx.x; i < nparts; i += 128)
	{
		val = cache_d[i];
		if (val > 0)
		{
			atomicAdd(&pwgts[i], val);
		}
	}
}

/*CUDA-init pwgts array*/
__global__ void inittpwgts(float *tpwgts, float temp, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nparts)
		tpwgts[ii] = temp;
}

/*Malloc initial partition phase to refine phase params*/
void Mallocinit_refineinfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;
	int num = 0;

	// printf("hunyuangraph_malloc_refineinfo nvtxs=%d\n", nvtxs);

	// cudaMalloc((void**)&graph->cuda_where,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bnd,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bndnum,sizeof(int));
	// cudaMalloc((void**)&graph->cuda_pwgts,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_tpwgts,nparts * sizeof(float));
	// cudaMalloc((void**)&graph->cuda_maxwgt,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_minwgt,nparts * sizeof(int));

	if(GPU_Memory_Pool)
	{
		// graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "Mallocinit_refineinfo: where");
		// graph->cuda_bnd = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "Mallocinit_refineinfo: bnd");
		graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: pwgts");
		graph->cuda_tpwgts = (float *)lmalloc_with_check(sizeof(float) * nparts, "Mallocinit_refineinfo: tpwgts");
		graph->cuda_maxwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: maxwgt");
		graph->cuda_minwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: minwgt");
		// graph->cuda_bndnum = (int *)lmalloc_with_check(sizeof(int), "Mallocinit_refineinfo: bndnum");

		graph->cuda_balance = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_malloc_refineinfo: cuda_balance");
		// graph->cuda_bn = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_bn");
		graph->cuda_select = (char *)lmalloc_with_check(sizeof(char) * graph->nvtxs,"cuda_select");
		graph->cuda_to = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_to");
		graph->cuda_gain = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_gain");
		// graph->cuda_csr = (int *)lmalloc_with_check(sizeof(int) * 2, "Mallocinit_refineinfo: cuda_csr");
		// graph->cuda_que = (int *)lmalloc_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "Mallocinit_refineinfo: cuda_que");
	}
	else
	{
		cudaMalloc((void **)&graph->cuda_pwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_tpwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_maxwgt, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_minwgt, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_balance, sizeof(int));
		cudaMalloc((void **)&graph->cuda_select, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_to, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_gain, sizeof(int) * graph->nvtxs);
	}

	cudaMemcpy(graph->cuda_where, graph->where, nvtxs * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(graph->cuda_bndnum, &num, sizeof(int), cudaMemcpyHostToDevice);

	initpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_pwgts, nparts);

	calculateSum<<<(nvtxs + 127) / 128, 128, nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_where, graph->cuda_vwgt);

	// inittpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_tpwgts, hunyuangraph_admin->tpwgts[0], nparts);
	cudaMemcpy(graph->cuda_tpwgts,hunyuangraph_admin->tpwgts,nparts * sizeof(float),cudaMemcpyHostToDevice);
}

/*Malloc refine params*/
void hunyuangraph_malloc_refineinfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;
	int num = 0;

	// printf("hunyuangraph_malloc_refineinfo nvtxs=%d\n", nvtxs);

	// cudaMalloc((void**)&graph->cuda_bnd,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bndnum,sizeof(int));
	// cudaMalloc((void**)&graph->cuda_pwgts,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_tpwgts,nparts * sizeof(float));
	// cudaMalloc((void**)&graph->cuda_maxwgt,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_minwgt,nparts * sizeof(int));

	if(GPU_Memory_Pool)
	{
		// graph->cuda_bnd = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_refineinfo: bnd");
		graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: pwgts");
		graph->cuda_tpwgts = (float *)lmalloc_with_check(sizeof(float) * nparts, "hunyuangraph_malloc_refineinfo: tpwgts");
		graph->cuda_maxwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: maxwgt");
		graph->cuda_minwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: minwgt");
		// graph->cuda_bndnum = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_malloc_refineinfo: bndnum");

		graph->cuda_balance = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_malloc_refineinfo: cuda_balance");
		// graph->cuda_bn = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_malloc_refineinfo: cuda_bn");
		graph->cuda_select = (char *)lmalloc_with_check(sizeof(char) * graph->nvtxs,"cuda_select");
		graph->cuda_to = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_to");
		graph->cuda_gain = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_gain");
		// graph->cuda_csr = (int *)lmalloc_with_check(sizeof(int) * 2, "hunyuangraph_malloc_refineinfo: cuda_csr");
		// graph->cuda_que = (int *)lmalloc_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "hunyuangraph_malloc_refineinfo: cuda_que");
	}
	else
	{
		cudaMalloc((void **)&graph->cuda_pwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_tpwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_maxwgt, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_minwgt, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_balance, sizeof(int));
		cudaMalloc((void **)&graph->cuda_select, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_to, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_gain, sizeof(int) * graph->nvtxs);
	}
	cudaMemcpy(graph->cuda_bndnum, &num, sizeof(int), cudaMemcpyHostToDevice);

	initpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_pwgts, nparts);

	calculateSum<<<(nvtxs + 127) / 128, 128, nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_where, graph->cuda_vwgt);

	// inittpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_tpwgts, hunyuangraph_admin->tpwgts[0], nparts);
	cudaMemcpy(graph->cuda_tpwgts,hunyuangraph_admin->tpwgts,nparts * sizeof(float),cudaMemcpyHostToDevice);
}

/*CUDA-kway parjection*/
__global__ void projectback(int *where, int *cwhere, int *cmap, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int t = cmap[ii];
		where[ii] = cwhere[t];
	}
}

/*Kway parjection*/
void hunyuangraph_kway_project(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	hunyuangraph_graph_t *cgraph = graph->coarser;

	projectback<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_where, cgraph->cuda_where, graph->cuda_cmap, nvtxs);
}

/*Free graph uncoarsening phase params*/
void hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	if(GPU_Memory_Pool)
	{
		// printf("hunyuangraph_uncoarsen_free_krefine nvtxs=%d\n", graph->nvtxs);
		// lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "hunyuangraph_uncoarsen_free_krefine: cuda_que");	// cuda_que
		// lfree_with_check(sizeof(int) * 2, "cuda_csr");																	 	// cuda_csr
		lfree_with_check((void *)graph->cuda_gain, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_gain");					// cuda_gain
		lfree_with_check((void *)graph->cuda_to, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_to");						// cuda_to
		lfree_with_check((void *)graph->cuda_select, sizeof(char) * graph->nvtxs, "cuda_select");													// cuda_select
		// lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_bn");					 	// cuda_bn
		lfree_with_check((void *)graph->cuda_balance, sizeof(int), "hunyuangraph_uncoarsen_free_krefine: cuda_balance");							// cuda_balance
		// lfree_with_check(sizeof(int), "hunyuangraph_uncoarsen_free_krefine: bndnum");									 	// bndnum
		lfree_with_check((void *)graph->cuda_minwgt, sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: minwgt");		// minwgt
		lfree_with_check((void *)graph->cuda_maxwgt, sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: maxwgt");		// maxwgt
		lfree_with_check((void *)graph->cuda_tpwgts, sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: tpwgts");		// tpwgts
		lfree_with_check((void *)graph->cuda_pwgts, sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: pwgts");		// pwgts
		// lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: bnd");						 	// bnd
	}
	else
	{
		cudaFree(graph->cuda_gain);
		cudaFree(graph->cuda_to);
		cudaFree(graph->cuda_select);
		cudaFree(graph->cuda_balance);
		cudaFree(graph->cuda_minwgt);
		cudaFree(graph->cuda_maxwgt);
		cudaFree(graph->cuda_tpwgts);
		cudaFree(graph->cuda_pwgts);
	}
}

void hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	// printf("hunyuangraph_uncoarsen_free_coarsen nvtxs=%d\n", graph->nvtxs);
	if(GPU_Memory_Pool)
	{
		lfree_with_check((void *)graph->cuda_where, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: where");						// where
		if (graph->cuda_cmap != NULL)
			lfree_with_check((void *)graph->cuda_cmap, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: cmap");					// cmap;
		if (graph->bin_idx != NULL)
			lfree_with_check((void *)graph->bin_idx, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: bin_idx");					//	bin_idx
		if (graph->bin_offset != NULL)
			lfree_with_check((void *)graph->bin_offset, sizeof(int) * 15, "hunyuangraph_uncoarsen_free_coarsen: bin_offset");						//	bin_offset
		if (graph->length_vertex != NULL)
			lfree_with_check((void *)graph->length_vertex, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: length_vertex");		//	length_vertex
		lfree_with_check((void *)graph->cuda_adjwgt, sizeof(int) * graph->nedges, "hunyuangraph_uncoarsen_free_coarsen: adjwgt");					// adjwgt
		lfree_with_check((void *)graph->cuda_adjncy, sizeof(int) * graph->nedges, "hunyuangraph_uncoarsen_free_coarsen: adjncy");					// adjncy
		lfree_with_check((void *)graph->cuda_xadj, sizeof(int) * (graph->nvtxs + 1), "hunyuangraph_uncoarsen_free_coarsen: xadj");					// xadj
		lfree_with_check((void *)graph->cuda_vwgt, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: vwgt");						// vwgt
	}
	else
	{
		cudaFree(graph->cuda_where);
		if (graph->cuda_cmap != NULL)
			cudaFree(graph->cuda_cmap);
		cudaFree(graph->bin_idx);
		cudaFree(graph->bin_offset);
		cudaFree(graph->length_vertex);
		cudaFree(graph->cuda_adjwgt);
		cudaFree(graph->cuda_adjncy);
		cudaFree(graph->cuda_xadj);
		cudaFree(graph->cuda_vwgt);
	}

	if(graph->h_bin_offset != NULL)
		free(graph->h_bin_offset);
}

void hunyuangraph_GPU_uncoarsen(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, hunyuangraph_graph_t *cgraph)
{
	Mallocinit_refineinfo(hunyuangraph_admin, cgraph);

	hunyuangraph_k_refinement(hunyuangraph_admin, cgraph);
	// hunyuangraph_k_refinement_me(hunyuangraph_admin,cgraph);

	hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin, cgraph);

	for (int i = 0;; i++)
	{
		if (cgraph != graph)
		{
			cgraph = cgraph->finer;

			// cudaMalloc((void**)&cgraph->cuda_where, cgraph->nvtxs * sizeof(int));

			hunyuangraph_kway_project(hunyuangraph_admin, cgraph);

			hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin, cgraph->coarser);

			hunyuangraph_malloc_refineinfo(hunyuangraph_admin, cgraph);

			hunyuangraph_k_refinement(hunyuangraph_admin, cgraph);
			// hunyuangraph_k_refinement_me(hunyuangraph_admin,cgraph);

			hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin, cgraph);

			// hunyuangraph_free_uncoarsen(hunyuangraph_admin, cgraph->coarser);
		}
		else
			break;
	}
}

void hunyuangraph_GPU_uncoarsen_SC25(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, hunyuangraph_graph_t *cgraph, int *level)
{
	Mallocinit_refineinfo(hunyuangraph_admin, cgraph);

	hunyuangraph_k_refinement_SC25(hunyuangraph_admin, cgraph, level);
	// hunyuangraph_k_refinement_me(hunyuangraph_admin,cgraph);

	hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin, cgraph);

	for (int i = 0;; i++)
	{
		if (cgraph != graph)
		{
			cgraph = cgraph->finer;
			level[0]--;

			// cudaMalloc((void**)&cgraph->cuda_where, cgraph->nvtxs * sizeof(int));

			hunyuangraph_kway_project(hunyuangraph_admin, cgraph);

			hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin, cgraph->coarser);

			hunyuangraph_malloc_refineinfo(hunyuangraph_admin, cgraph);

			hunyuangraph_k_refinement_SC25(hunyuangraph_admin, cgraph, level);
			// hunyuangraph_k_refinement_me(hunyuangraph_admin,cgraph);

			hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin, cgraph);

			// hunyuangraph_free_uncoarsen(hunyuangraph_admin, cgraph->coarser);
		}
		else
			break;
	}
}

#endif