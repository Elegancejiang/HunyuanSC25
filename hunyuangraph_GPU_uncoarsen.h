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

	printf("hunyuangraph_malloc_refineinfo nvtxs=%d\n", nvtxs);

	// cudaMalloc((void**)&graph->cuda_where,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bnd,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bndnum,sizeof(int));
	// cudaMalloc((void**)&graph->cuda_pwgts,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_tpwgts,nparts * sizeof(float));
	// cudaMalloc((void**)&graph->cuda_maxwgt,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_minwgt,nparts * sizeof(int));

	graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "Mallocinit_refineinfo: where");
	graph->cuda_bnd = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "Mallocinit_refineinfo: bnd");
	graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: pwgts");
	graph->cuda_tpwgts = (float *)lmalloc_with_check(sizeof(float) * nparts, "Mallocinit_refineinfo: tpwgts");
	graph->cuda_maxwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: maxwgt");
	graph->cuda_minwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: minwgt");
	graph->cuda_bndnum = (int *)lmalloc_with_check(sizeof(int), "Mallocinit_refineinfo: bndnum");

	graph->cuda_bn = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_bn");
	graph->cuda_bt = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_bt");
	graph->cuda_g = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_g");
	graph->cuda_csr = (int *)lmalloc_with_check(sizeof(int) * 2, "Mallocinit_refineinfo: cuda_csr");
	graph->cuda_que = (int *)lmalloc_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "Mallocinit_refineinfo: cuda_que");

	cudaMemcpy(graph->cuda_where, graph->where, nvtxs * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(graph->cuda_bndnum, &num, sizeof(int), cudaMemcpyHostToDevice);

	initpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_pwgts, nparts);

	calculateSum<<<(nvtxs + 127) / 128, 128, nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_where, graph->cuda_vwgt);

	inittpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_tpwgts, hunyuangraph_admin->tpwgts[0], nparts);
}

/*Malloc refine params*/
void hunyuangraph_malloc_refineinfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;
	int num = 0;

	printf("hunyuangraph_malloc_refineinfo nvtxs=%d\n", nvtxs);

	// cudaMalloc((void**)&graph->cuda_bnd,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bndnum,sizeof(int));
	// cudaMalloc((void**)&graph->cuda_pwgts,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_tpwgts,nparts * sizeof(float));
	// cudaMalloc((void**)&graph->cuda_maxwgt,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_minwgt,nparts * sizeof(int));

	graph->cuda_bnd = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_refineinfo: bnd");
	graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: pwgts");
	graph->cuda_tpwgts = (float *)lmalloc_with_check(sizeof(float) * nparts, "hunyuangraph_malloc_refineinfo: tpwgts");
	graph->cuda_maxwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: maxwgt");
	graph->cuda_minwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: minwgt");
	graph->cuda_bndnum = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_malloc_refineinfo: bndnum");

	graph->cuda_bn = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_malloc_refineinfo: cuda_bn");
	graph->cuda_bt = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_malloc_refineinfo: cuda_bt");
	graph->cuda_g = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_malloc_refineinfo: cuda_g");
	graph->cuda_csr = (int *)lmalloc_with_check(sizeof(int) * 2, "hunyuangraph_malloc_refineinfo: cuda_csr");
	graph->cuda_que = (int *)lmalloc_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "hunyuangraph_malloc_refineinfo: cuda_que");

	cudaMemcpy(graph->cuda_bndnum, &num, sizeof(int), cudaMemcpyHostToDevice);

	initpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_pwgts, nparts);

	calculateSum<<<(nvtxs + 127) / 128, 128, nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_where, graph->cuda_vwgt);

	inittpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_tpwgts, hunyuangraph_admin->tpwgts[0], nparts);
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
	printf("hunyuangraph_uncoarsen_free_krefine nvtxs=%d\n", graph->nvtxs);
	lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "hunyuangraph_uncoarsen_free_krefine: cuda_que"); // cuda_que
	lfree_with_check(sizeof(int) * 2, "cuda_csr");															 // cuda_csr
	lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_g");					 // cuda_g
	lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_bt");					 // cuda_bt
	lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_bn");					 // cuda_bn
	lfree_with_check(sizeof(int), "hunyuangraph_uncoarsen_free_krefine: bndnum");									 // bndnum
	lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: minwgt");		 // minwgt
	lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: maxwgt");		 // maxwgt
	lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: tpwgts");		 // tpwgts
	lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: pwgts");		 // pwgts
	lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: bnd");						 // bnd

	// cudaFree(graph->cuda_adjwgt);
	// cudaFree(graph->cuda_adjncy);
	// cudaFree(graph->cuda_xadj);
	// cudaFree(graph->cuda_vwgt);
	//   cudaFree(graph->cuda_cmap);
	//   cudaFree(graph->cuda_maxwgt);
	//   cudaFree(graph->cuda_minwgt);
	//   cudaFree(graph->cuda_where);
	//   cudaFree(graph->cuda_pwgts);
	//   cudaFree(graph->cuda_bnd);
	//   cudaFree(graph->cuda_bndnum);
	//   cudaFree(graph->cuda_real_bnd_num);
	//   cudaFree(graph->cuda_real_bnd);
	//   cudaFree(graph->cuda_tpwgts);
}

void hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	printf("hunyuangraph_uncoarsen_free_coarsen nvtxs=%d\n", graph->nvtxs);

	lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: where");		 // where
	if (graph->cuda_cmap != NULL)
		lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: cmap");	 // cmap;
	lfree_with_check(sizeof(int) * graph->nedges, "hunyuangraph_uncoarsen_free_coarsen: adjwgt");	 // adjwgt
	lfree_with_check(sizeof(int) * graph->nedges, "hunyuangraph_uncoarsen_free_coarsen: adjncy");	 // adjncy
	lfree_with_check(sizeof(int) * (graph->nvtxs + 1), "hunyuangraph_uncoarsen_free_coarsen: xadj"); // xadj
	lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: vwgt");		 // vwgt
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

#endif