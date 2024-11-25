#ifndef _H_GPU_COARSEN
#define _H_GPU_COARSEN

#include "hunyuangraph_struct.h"
#include "hunyuangraph_GPU_match.h"
#include "hunyuangraph_GPU_contraction.h"

/*Malloc gpu coarsen graph params*/
void hunyuangraph_malloc_coarseninfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
    int nvtxs = graph->nvtxs;
    int nedges = graph->nedges;

    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc, NULL);
    // cudaMalloc((void**)&graph->cuda_match,nvtxs * sizeof(int));
    graph->cuda_match = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "match");
    // cudaMalloc((void**)&graph->cuda_cmap,nvtxs * sizeof(int));
    graph->cuda_cmap = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "cmap");
    graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "where"); // 不可在k-way refinement再申请空间，会破坏栈的原则
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc, NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
}

void hunyuangraph_memcpy_coarsentoinit(hunyuangraph_graph_t *graph)
{
    int nvtxs = graph->nvtxs;
    int nedges = graph->nedges;

    graph->xadj = (int *)malloc(sizeof(int) * (nvtxs + 1));
    graph->vwgt = (int *)malloc(sizeof(int) * nvtxs);
    graph->adjncy = (int *)malloc(sizeof(int) * nedges);
    graph->adjwgt = (int *)malloc(sizeof(int) * nedges);

    cudaDeviceSynchronize();
    gettimeofday(&begin_coarsen_memcpy, NULL);
    cudaMemcpy(graph->xadj, graph->cuda_xadj, (nvtxs + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(graph->vwgt, graph->cuda_vwgt, nvtxs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(graph->adjncy, graph->cuda_adjncy, nedges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(graph->adjwgt, graph->cuda_adjwgt, nedges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&end_coarsen_memcpy, NULL);
    coarsen_memcpy += (end_coarsen_memcpy.tv_sec - begin_coarsen_memcpy.tv_sec) * 1000 + (end_coarsen_memcpy.tv_usec - begin_coarsen_memcpy.tv_usec) / 1000.0;
}

/*Gpu multilevel coarsen*/
hunyuangraph_graph_t *hunyuangarph_coarsen(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
    int level = 0;

    hunyuangraph_admin->maxvwgt = 1.5 * graph->tvwgt[0] / hunyuangraph_admin->Coarsen_threshold;

    // printf("level %2d: nvtxs %10d nedges %10d nedges/nvtxs=%7.2lf adjwgtsum %12d\n",level, graph->nvtxs, graph->nedges, (double)graph->nedges / (double)graph->nvtxs, compute_graph_adjwgtsum_gpu(graph));

    do
    {
        hunyuangraph_malloc_coarseninfo(hunyuangraph_admin, graph);

        cudaDeviceSynchronize();
        gettimeofday(&begin_part_match, NULL);
        hunyuangraph_graph_t *cgraph = hunyuangraph_gpu_match(hunyuangraph_admin, graph, level);
        cudaDeviceSynchronize();
        gettimeofday(&end_part_match, NULL);
        part_match += (end_part_match.tv_sec - begin_part_match.tv_sec) * 1000 + (end_part_match.tv_usec - begin_part_match.tv_usec) / 1000.0;

        cudaDeviceSynchronize();
        gettimeofday(&begin_part_contruction, NULL);
        hunyuangraph_gpu_create_cgraph(hunyuangraph_admin, graph, cgraph);
        cudaDeviceSynchronize();
        gettimeofday(&end_part_contruction, NULL);
        part_contruction += (end_part_contruction.tv_sec - begin_part_contruction.tv_sec) * 1000 + (end_part_contruction.tv_usec - begin_part_contruction.tv_usec) / 1000.0;

        graph = graph->coarser;
        level++;

        // printf("level %2d: nvtxs %10d nedges %10d nedges/nvtxs=%7.2lf adjwgtsum %12d\n", level, graph->nvtxs, graph->nedges, (double)graph->nedges / (double)graph->nvtxs, compute_graph_adjwgtsum_gpu(graph));

    } while (
        graph->nvtxs > hunyuangraph_admin->Coarsen_threshold &&
        graph->nvtxs < 0.85 * graph->finer->nvtxs &&
        graph->nedges > graph->nvtxs / 2);

    hunyuangraph_memcpy_coarsentoinit(graph);

    return graph;
}

#endif