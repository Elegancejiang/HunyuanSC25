#ifndef _H_PARTITIONGRAPH
#define _H_PARTITIONGRAPH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_timer.h"
#include "hunyuangraph_GPU_memory.h"
#include "hunyuangraph_GPU_coarsen.h"
#include "hunyuangraph_initialpartition.h"
#include "hunyuangraph_GPU_uncoarsen.h"

/*Graph kway-partition algorithm*/
void hunyuangraph_kway_partition(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int *part)
{
	hunyuangraph_graph_t *cgraph;

	cudaDeviceSynchronize();
	gettimeofday(&begin_part_coarsen, NULL);
	cgraph = hunyuangarph_coarsen(hunyuangraph_admin, graph);
	cudaDeviceSynchronize();
	gettimeofday(&end_part_coarsen, NULL);
	part_coarsen += (end_part_coarsen.tv_sec - begin_part_coarsen.tv_sec) * 1000 + (end_part_coarsen.tv_usec - begin_part_coarsen.tv_usec) / 1000.0;

	printf("Coarsen end:cnvtxs=%d cnedges=%d\n", cgraph->nvtxs, cgraph->nedges);

	cudaDeviceSynchronize();
	gettimeofday(&begin_part_init, NULL);
	hunyuangarph_initialpartition(hunyuangraph_admin, cgraph);
	cudaDeviceSynchronize();
	gettimeofday(&end_part_init, NULL);
	part_init += (end_part_init.tv_sec - begin_part_init.tv_sec) * 1000 + (end_part_init.tv_usec - begin_part_init.tv_usec) / 1000.0;

	// hunyuangraph_memcpy_coarsentoinit(cgraph);
	// int e = hunyuangraph_computecut(cgraph, cgraph->where);
	// printf("edgecut=%d\n",e);
	// printf("Init partition end\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_part_uncoarsen, NULL);
	hunyuangraph_GPU_uncoarsen(hunyuangraph_admin, graph, cgraph);
	cudaDeviceSynchronize();
	gettimeofday(&end_part_uncoarsen, NULL);
	part_uncoarsen += (end_part_uncoarsen.tv_sec - begin_part_uncoarsen.tv_sec) * 1000 + (end_part_uncoarsen.tv_usec - begin_part_uncoarsen.tv_usec) / 1000.0;

	// printf("Uncoarsen end\n");
}

/*Set kway balance params*/
void hunyuangraph_set_kway_bal(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	for (int i = 0, j = 0; i < hunyuangraph_admin->nparts; i++)
		hunyuangraph_admin->part_balance[i + j] = graph->tvwgt_reverse[j] / hunyuangraph_admin->tpwgts[i + j];
}

__global__ void init_vwgt(int *vwgt, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
		vwgt[ii] = 1;
}

__global__ void init_adjwgt(int *adjwgt, int nedges)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nedges)
		adjwgt[ii] = 1;
}

/*Malloc and memcpy original graph from cpu to gpu*/
void hunyuangraph_malloc_original_coarseninfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nedges = graph->nedges;
	graph->cuda_vwgt = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "vwgt");
	graph->cuda_xadj = (int *)lmalloc_with_check(sizeof(int) * (nvtxs + 1), "xadj");
	graph->cuda_adjncy = (int *)lmalloc_with_check(sizeof(int) * nedges, "adjncy");
	graph->cuda_adjwgt = (int *)lmalloc_with_check(sizeof(int) * nedges, "adjwgt");

	// cudaMalloc((void**)&graph->cuda_xadj,(nvtxs+1)*sizeof(int));
	// cudaMalloc((void**)&graph->cuda_vwgt,nvtxs*sizeof(int));
	// cudaMalloc((void**)&graph->cuda_adjncy,nedges*sizeof(int));
	// cudaMalloc((void**)&graph->cuda_adjwgt,nedges*sizeof(int));

	cudaMemcpy(graph->cuda_xadj, graph->xadj, (nvtxs + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(graph->cuda_adjncy, graph->adjncy, nedges * sizeof(int), cudaMemcpyHostToDevice);

	// ����CUDA��
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// ���õ�һ���˺���
	init_vwgt<<<(nvtxs + 127) / 128, 128, 0, stream>>>(graph->cuda_vwgt, nvtxs);
	// ���õڶ����˺���
	init_adjwgt<<<(nedges + 127) / 128, 128, 0, stream>>>(graph->cuda_adjwgt, nedges);

	// �ȴ����������??
	cudaStreamSynchronize(stream);

	// ����CUDA��
	cudaStreamDestroy(stream);

	// cudaMemcpy(graph->cuda_vwgt,graph->vwgt,nvtxs*sizeof(int),cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_adjwgt,graph->adjwgt,nedges*sizeof(int),cudaMemcpyHostToDevice);
}

/*Graph partition algorithm*/
void hunyuangraph_PartitionGraph(int *nvtxs, int *xadj, int *adjncy, int *vwgt, int *adjwgt, int *nparts, float *tpwgts, float *ubvec, int *part)
{
	hunyuangraph_graph_t *graph;
	hunyuangraph_admin_t *hunyuangraph_admin;

	hunyuangraph_admin = hunyuangraph_set_graph_admin(*nparts, tpwgts, ubvec);

	graph = hunyuangraph_set_first_level_graph(*nvtxs, xadj, adjncy, vwgt, adjwgt);

	hunyuangraph_set_kway_bal(hunyuangraph_admin, graph);

	hunyuangraph_admin->Coarsen_threshold = hunyuangraph_max((*nvtxs) / (20 * (hunyuangraph_compute_log2(*nparts))), 30 * (*nparts));
	hunyuangraph_admin->nIparts = (hunyuangraph_admin->Coarsen_threshold == 30 * (*nparts) ? 4 : 5);

	/*hunyuangraph_admin->Coarsen_threshold = (*nparts) * 8;
	if (hunyuangraph_admin->Coarsen_threshold > 1024)
	{
		hunyuangraph_admin->Coarsen_threshold = (*nparts) * 2;
		hunyuangraph_admin->Coarsen_threshold = hunyuangraph_max(1024, hunyuangraph_admin->Coarsen_threshold);
	}
	printf("hunyuangraph_admin->Coarsen_threshold=%10d\n", hunyuangraph_admin->Coarsen_threshold);*/

	Malloc_GPU_Memory(graph->nvtxs, graph->nedges);

	// cudaMalloc((void**)&cu_bn, graph->nvtxs * sizeof(int));
	// cudaMalloc((void**)&cu_bt, graph->nvtxs * sizeof(int));
	// cudaMalloc((void**)&cu_g,  graph->nvtxs * sizeof(int));
	// cudaMalloc((void**)&cu_csr, 2 * sizeof(int));
	// cudaMalloc((void**)&cu_que, 2 * hunyuangraph_admin->nparts * sizeof(int));

	// cu_bn  = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs,"cu_bn");
	// cu_bt  = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs,"cu_bt");
	// cu_g   = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs,"cu_g");
	// cu_csr = (int *)lmalloc_with_check(sizeof(int) * 2,"cu_csr");
	// cu_que = (int *)lmalloc_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2,"cu_que");

	hunyuangraph_malloc_original_coarseninfo(hunyuangraph_admin, graph);

	printf("begin partition\n");
	cudaDeviceSynchronize();
	gettimeofday(&begin_part_all, NULL);
	hunyuangraph_kway_partition(hunyuangraph_admin, graph, part);
	cudaDeviceSynchronize();
	gettimeofday(&end_part_all, NULL);
	part_all += (end_part_all.tv_sec - begin_part_all.tv_sec) * 1000 + (end_part_all.tv_usec - begin_part_all.tv_usec) / 1000.0;
	printf("end partition\n");

	cudaMemcpy(part, graph->cuda_where, graph->nvtxs * sizeof(int), cudaMemcpyDeviceToHost);

	hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin, graph);

	// lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2,"cu_que");	//cu_que
	// lfree_with_check(sizeof(int) * 2,"cu_csr");									//cu_csr
	// lfree_with_check(sizeof(int) * graph->nvtxs,"cu_g");						//cu_g
	// lfree_with_check(sizeof(int) * graph->nvtxs,"cu_bt");						//cu_bt
	// lfree_with_check(sizeof(int) * graph->nvtxs,"cu_bn");						//cu_bn

	Free_GPU_Memory();
}

#endif