#ifndef _H_GRAPH
#define _H_GRAPH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_GPU_memory.h"

/*Set graph params*/
void hunyuangraph_init_cpu_graph(hunyuangraph_graph_t *graph)
{
  memset((void *)graph, 0, sizeof(hunyuangraph_graph_t));
  graph->nvtxs = -1;
  graph->nedges = -1;
  graph->xadj = NULL;
  graph->vwgt = NULL;
  graph->adjncy = NULL;
  graph->adjwgt = NULL;
  graph->label = NULL;
  graph->cmap = NULL;
  graph->tvwgt = NULL;
  graph->tvwgt_reverse = NULL;
  graph->where = NULL;
  graph->pwgts = NULL;
  graph->mincut = -1;
  graph->nbnd = -1;
  graph->id = NULL;
  graph->ed = NULL;
  graph->bndptr = NULL;
  graph->bndlist = NULL;
  graph->coarser = NULL;
  graph->finer = NULL;
}

/*Malloc graph*/
hunyuangraph_graph_t *hunyuangraph_create_cpu_graph(void)
{
  hunyuangraph_graph_t *graph = (hunyuangraph_graph_t *)malloc(sizeof(hunyuangraph_graph_t));
  hunyuangraph_init_cpu_graph(graph);
  return graph;
}

/*Set graph tvwgt value*/
void hunyuangraph_set_graph_tvwgt(hunyuangraph_graph_t *graph)
{
  if (graph->tvwgt == NULL)
  {
    graph->tvwgt = (int *)malloc(sizeof(int));
  }

  if (graph->tvwgt_reverse == NULL)
  {
    graph->tvwgt_reverse = (float *)malloc(sizeof(float));
  }

  graph->tvwgt[0] = hunyuangraph_int_sum(graph->nvtxs, graph->vwgt);
  graph->tvwgt_reverse[0] = 1.0 / (graph->tvwgt[0] > 0 ? graph->tvwgt[0] : 1);
}

/*Set graph vertex label*/
void hunyuangraph_set_graph_label(hunyuangraph_graph_t *graph)
{
  if (graph->label == NULL)
  {
    graph->label = (int *)malloc(sizeof(int) * (graph->nvtxs));
  }

  for (int i = 0; i < graph->nvtxs; i++)
    graph->label[i] = i;
}

/*Set graph information*/
hunyuangraph_graph_t *hunyuangraph_set_graph(hunyuangraph_admin_t *hunyuangraph_admin, int nvtxs, int *xadj, int *adjncy, int *vwgt, int *adjwgt, int *tvwgt)
{
  hunyuangraph_graph_t *graph = hunyuangraph_create_cpu_graph();

  graph->nvtxs = nvtxs;
  graph->nedges = xadj[nvtxs];
  graph->xadj = xadj;
  graph->adjncy = adjncy;

  graph->vwgt = vwgt;
  graph->adjwgt = adjwgt;

  graph->tvwgt = (int *)malloc(sizeof(int));
  graph->tvwgt_reverse = (float *)malloc(sizeof(float));

  graph->tvwgt[0] = tvwgt[0];
  graph->tvwgt_reverse[0] = 1.0 / (graph->tvwgt[0] > 0 ? graph->tvwgt[0] : 1);

  // init label spend much time
  //  graph->label = (int *)malloc(sizeof(int) * (graph->nvtxs));
  //  for(int i = 0;i < graph->nvtxs;i++)
  //      graph->label[i] = i;

  return graph;
}

hunyuangraph_graph_t *hunyuangraph_set_first_level_graph(int nvtxs, int *xadj, int *adjncy, int *vwgt, int *adjwgt)
{
  int i;
  hunyuangraph_graph_t *graph;

  graph = hunyuangraph_create_cpu_graph();
  graph->nvtxs = nvtxs;
  graph->nedges = xadj[nvtxs];
  graph->xadj = xadj;
  graph->adjncy = adjncy;

  graph->vwgt = vwgt;
  graph->adjwgt = adjwgt;

  graph->tvwgt = (int *)malloc(sizeof(int));
  graph->tvwgt_reverse = (float *)malloc(sizeof(float));
  graph->tvwgt[0] = nvtxs;
  graph->tvwgt_reverse[0] = 1.0 / (graph->tvwgt[0] > 0 ? graph->tvwgt[0] : 1);

  return graph;
}

/*Compute Partition result edge-cut*/
int hunyuangraph_computecut(hunyuangraph_graph_t *graph, int *where)
{
  int i, j, cut = 0;
  for (i = 0; i < graph->nvtxs; i++)
  {
    // printf("i=%d\n",i);
    for (j = graph->xadj[i]; j < graph->xadj[i + 1]; j++)
      if (where[i] != where[graph->adjncy[j]])
        cut += graph->adjwgt[j];
  }
  return cut / 2;
}

int compute_graph_adjwgtsum_cpu(hunyuangraph_graph_t *graph)
{
  int sum = 0;
  for (int i = 0; i < graph->nedges; i++)
    sum += graph->adjwgt[i];
  return sum;
}

int compute_graph_adjwgtsum_gpu(hunyuangraph_graph_t *graph)
{
  int sum = thrust::reduce(thrust::device, graph->cuda_adjwgt, graph->cuda_adjwgt + graph->nedges);

  return sum;
}

/*Malloc cpu coarsen graph params*/
hunyuangraph_graph_t *hunyuangraph_set_cpu_cgraph(hunyuangraph_graph_t *graph, int cnvtxs)
{
  hunyuangraph_graph_t *cgraph;
  cgraph = hunyuangraph_create_cpu_graph();

  cgraph->nvtxs = cnvtxs;
  cgraph->xadj = (int *)malloc(sizeof(int) * (cnvtxs + 1));
  cgraph->adjncy = (int *)malloc(sizeof(int) * (graph->nedges));
  cgraph->adjwgt = (int *)malloc(sizeof(int) * (graph->nedges));
  cgraph->vwgt = (int *)malloc(sizeof(int) * cnvtxs);
  cgraph->tvwgt = (int *)malloc(sizeof(int));
  cgraph->tvwgt_reverse = (float *)malloc(sizeof(float));

  cgraph->finer = graph;
  graph->coarser = cgraph;

  return cgraph;
}

/*Malloc gpu coarsen graph params*/
hunyuangraph_graph_t *hunyuangraph_set_gpu_cgraph(hunyuangraph_graph_t *graph, int cnvtxs)
{
  hunyuangraph_graph_t *cgraph = hunyuangraph_create_cpu_graph();

  cgraph->nvtxs = cnvtxs;
  //   cgraph->xadj=(int*)malloc(sizeof(int)*(cnvtxs+1));
  cgraph->tvwgt = (int *)malloc(sizeof(int));
  cgraph->tvwgt_reverse = (float *)malloc(sizeof(float));

  cgraph->finer = graph;
  graph->coarser = cgraph;

  return cgraph;
}

/*Set split graph params*/
hunyuangraph_graph_t *hunyuangraph_set_splitgraph(hunyuangraph_graph_t *graph, int snvtxs, int snedges)
{
  hunyuangraph_graph_t *sgraph;
  sgraph = hunyuangraph_create_cpu_graph();

  sgraph->nvtxs = snvtxs;
  sgraph->nedges = snedges;

  sgraph->xadj = (int *)malloc(sizeof(int) * (snvtxs + 1));
  sgraph->vwgt = (int *)malloc(sizeof(int) * (snvtxs + 1));
  sgraph->adjncy = (int *)malloc(sizeof(int) * (snedges));
  sgraph->adjwgt = (int *)malloc(sizeof(int) * (snedges));
  sgraph->label = (int *)malloc(sizeof(int) * (snvtxs));
  sgraph->tvwgt = (int *)malloc(sizeof(int));
  sgraph->tvwgt_reverse = (float *)malloc(sizeof(float));

  return sgraph;
}

/*Free graph params*/
void hunyuangraph_free_graph(hunyuangraph_graph_t **r_graph)
{
  hunyuangraph_graph_t *graph;
  graph = *r_graph;

  free(graph->xadj);
  free(graph->vwgt);
  free(graph->adjncy);
  free(graph->adjwgt);
  free(graph->where);
  free(graph->pwgts);
  free(graph->id);
  free(graph->ed);
  free(graph->bndptr);
  free(graph->bndlist);
  free(graph->tvwgt);
  free(graph->tvwgt_reverse);
  free(graph->label);
  free(graph->cmap);
  free(graph);
  *r_graph = NULL;
}

__global__ void exam_csr(int nvtxs, int *xadj, int *adjncy, int *adjwgt)
{
  for (int i = 0; i <= nvtxs && i < 200; i++)
    printf("%d ", xadj[i]);

  printf("\nadjncy/adjwgt:\n");
  for (int i = 0; i < nvtxs && i < 200; i++)
  {
    for (int j = xadj[i]; j < xadj[i + 1]; j++)
      printf("%d ", adjncy[j]);
    printf("\n");
    for (int j = xadj[i]; j < xadj[i + 1]; j++)
      printf("%d ", adjwgt[j]);
    printf("\n");
  }
  // printf("\n");
  // for (int i = 0; i < nvtxs; i++)
  // {
  // 	for (int j = xadj[i]; j < xadj[i + 1]; j++)
  // 		printf("%d ", adjwgt[j]);
  // 	printf("\n");
  // }
  // printf("\n");
}

#endif