#ifndef _H_GPU_SPLITGRAPH
#define _H_GPU_SPLITGRAPH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
__global__ void compute_lnedges(int nvtxs, int *xadj, int *adjncy, int *where, int *ltxadj)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 0)
		{
			int i, j, l, begin, end;
			begin = xadj[ii];
			end   = xadj[ii + 1];
			l     = 0;

			for(i = begin;i < end;i++)
			{
				j = adjncy[i];
				if(where[j] == 0) l ++;
			}
			ltxadj[ii] = l;
		}
		else
		{
			ltxadj[ii] = 0;
		}
	}
}

__global__ void compute_rnedges(int nvtxs, int *xadj, int *adjncy, int *where, int *rtxadj)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		// printf("ii=%d 1\n",ii);
		if(where[ii] == 1)
		{
			// printf("ii=%d 2\n",ii);
			int i, j, l, begin, end;
			begin = xadj[ii];
			end   = xadj[ii + 1];
			l     = 0;

			// printf("ii=%d 3\n",ii);
			for(i = begin;i < end;i++)
			{
				j = adjncy[i];
				if(where[j] == 1) l ++;
			}
			rtxadj[ii] = l;
		}
		else
		{
			rtxadj[ii] = 0;
		}
		// printf("ii=%d rtxadj=%d\n",ii,rtxadj[ii]);
	}
}

__global__ void compute_lmap(int nvtxs, int *where, int *lmap)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 0) lmap[ii] = 1;
		else lmap[ii] = 0;
	}
}

__global__ void compute_rmap(int nvtxs, int *where, int *rmap)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 1) rmap[ii] = 1;
		else rmap[ii] = 0;
	}
}

__global__ void set_lxadj(int nvtxs, int lnvtxs, int lnedges, int *xadj, int *where, int *lmap, int *ltxadj, int *lxadj)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 0) 
		{
			int val, ptr;
			val = ltxadj[ii];
			ptr = lmap[ii] - 1;
			lxadj[ptr] = val;
		}
	}
	else if(ii == nvtxs) lxadj[lnvtxs] = lnedges;
}

__global__ void set_rxadj(int nvtxs, int rnvtxs, int rnedges, int *xadj, int *where, int *rmap, int *rtxadj, int *rxadj)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 1) 
		{
			int val, ptr;
			val = rtxadj[ii];
			ptr = rmap[ii] - 1;
			rxadj[ptr] = val;
		}
	}
	else if(ii == nvtxs) rxadj[rnvtxs] = rnedges;
}

__global__ void set_ladjncy_ladjwgt(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *lxadj, int *ladjncy,\
	int *ladjwgt, int *lmap)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 0)
		{
			int i, j, begin, end, ptr;
			begin = xadj[ii];
			end   = xadj[ii + 1];
			ptr   = lmap[ii] - 1;
			ptr   = lxadj[ptr];

			for(i = begin;i < end;i++)
			{
				j = adjncy[i];
				if(where[j] == 0) 
				{
					ladjncy[ptr] = lmap[j] - 1;
					ladjwgt[ptr] = adjwgt[i];
					ptr++;
				}
			}
		}
	}
}

__global__ void set_radjncy_radjwgt(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *rxadj, int *radjncy,\
	int *radjwgt, int *rmap)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 1)
		{
			int i, j, begin, end, ptr;
			begin = xadj[ii];
			end   = xadj[ii + 1];
			ptr   = rmap[ii] - 1;
			ptr   = rxadj[ptr];

			for(i = begin;i < end;i++)
			{
				j = adjncy[i];
				if(where[j] == 1) 
				{
					radjncy[ptr] = rmap[j] - 1;
					radjwgt[ptr] = adjwgt[i];
					ptr++;
				}
			}
		}
	}
}

__global__ void set_lvwgt(int nvtxs, int *where, int *vwgt, int *lmap, int *lvwgt)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && where[ii] == 0)
	{
		int ptr = lmap[ii] - 1;
		lvwgt[ptr] = vwgt[ii];
	}
}

__global__ void set_rvwgt(int nvtxs, int *where, int *vwgt, int *rmap, int *rvwgt)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && where[ii] == 1)
	{
		int ptr = rmap[ii] - 1;
		rvwgt[ptr] = vwgt[ii];
	}
}

__global__ void set_llabel0(int nvtxs, int *where, int *lmap, int *llabel)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && where[ii] == 0)
	{
		int ptr = lmap[ii] - 1;
		llabel[ptr] = ii;
	}
}

__global__ void set_rlabel0(int nvtxs, int *where, int *lmap, int *rlabel)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && where[ii] == 1)
	{
		int ptr = lmap[ii] - 1;
		rlabel[ptr] = ii;
	}
}

__global__ void set_llabel1(int nvtxs, int *where, int *lmap, int *label, int *llabel)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && where[ii] == 0)
	{
		int ptr = lmap[ii] - 1;
		llabel[ptr] = label[ii];
	}
}

__global__ void set_rlabel1(int nvtxs, int *where, int *lmap, int *label, int *rlabel)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && where[ii] == 1)
	{
		int ptr = lmap[ii] - 1;
		rlabel[ptr] = label[ii];
	}
}

void set_subgraph_tvwgt(hunyuangraph_graph_t *graph, hunyuangraph_graph_t *lgraph, hunyuangraph_graph_t *rgraph)
{
	if(lgraph->tvwgt == NULL)
        lgraph->tvwgt = (int*)malloc(sizeof(int));
    if(lgraph->tvwgt_reverse == NULL)
        lgraph->tvwgt_reverse = (float*)malloc(sizeof(float));

	if(rgraph->tvwgt == NULL)
        rgraph->tvwgt = (int*)malloc(sizeof(int));
    if(rgraph->tvwgt_reverse == NULL)
        rgraph->tvwgt_reverse = (float*)malloc(sizeof(float));

	lgraph->tvwgt[0] = graph->pwgts[0];
	lgraph->tvwgt_reverse[0] = 1.0 / (lgraph->tvwgt[0] > 0 ? lgraph->tvwgt[0] : 1);

	rgraph->tvwgt[0] = graph->pwgts[1];
	rgraph->tvwgt_reverse[0] = 1.0 / (rgraph->tvwgt[0] > 0 ? rgraph->tvwgt[0] : 1);
}

__global__ void exam_temp_scan(int nvtxs, int *xadj, int *adjncy, int *where, int *temp_scan)
{
	for(int i = 0;i < nvtxs;i++)
	{
		for(int j = xadj[i];j < xadj[i + 1];j++)
			printf("%d ",adjncy[j]);
		printf("\n");
		for(int j = xadj[i];j < xadj[i + 1];j++)
			printf("%d ",where[adjncy[j]]);
		printf("\n");
		for(int j = xadj[i];j < xadj[i + 1];j++)
			printf("%d ",temp_scan[j]);
		printf("\n");
	}
}

__global__ void exam_lmap(int nvtxs, int *where, int *lmap)
{
	for(int i = 0;i < nvtxs;i++)
	{
		printf("ii=%d where=%d lmap=%d\n",i,where[i],lmap[i] - 1);
	}
}

__global__ void set_where(int *where)
{
	where[0] = 0;
	where[1] = 0;
	where[2] = 1;
	where[3] = 0;
	where[4] = 1;
}

__global__ void exam_graph(int nvtxs, int nedges, int *xadj, int *adjncy, int *vwgt, int *label)
{
	printf("lgraph->nvtxs:%d lgraph->nedges:%d\n",nvtxs,nedges);
  	printf("lgraph->vwgt:\n");
	for(int i = 0;i < nvtxs;i++)
	{
		printf("%d ",vwgt[i]);
	}
	printf("\nlgraph->label:\n");
	for(int i = 0;i < nvtxs;i++)
	{
		printf("%d ",label[i]);
	}
	printf("\nlgraph->xadj:\n");
	for(int i = 0;i <= nvtxs;i++)
	{
		printf("%d ",xadj[i]);
	}
	printf("\nlgraph->adjncy:\n");
	for(int i = 0;i < nvtxs;i++)
	{
		for(int j = xadj[i];j < xadj[i + 1];j++)
		{
			printf("%d ",adjncy[j]);
		}
		printf("\n");
	}
}

void splitgraph_GPU(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, \
	hunyuangraph_graph_t **r_lgraph, hunyuangraph_graph_t **r_rgraph, int flag)
{
	int nvtxs, nedges;

	hunyuangraph_graph_t *lgraph,*rgraph;
	lgraph=hunyuangraph_create_cpu_graph();
	rgraph=hunyuangraph_create_cpu_graph();

	int lnvtxs, rnvtxs, lnedges, rnedges;
	int *lmap, *rmap, *ltxadj, *rtxadj;

	nvtxs  = graph->nvtxs;
	nedges = graph->nedges;

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	// cudaMalloc((void**)&graph->cuda_xadj,sizeof(int) * (graph->nvtxs + 1));
	// cudaMalloc((void**)&graph->cuda_adjncy,sizeof(int) * graph->nedges);
	// cudaMalloc((void**)&graph->cuda_adjwgt,sizeof(int) * graph->nedges);
	// cudaMalloc((void**)&graph->cuda_vwgt,sizeof(int) * graph->nvtxs);
	// cudaMalloc((void**)&graph->cuda_where,sizeof(int) * graph->nvtxs);
	cudaMalloc((void**)&lmap,sizeof(int) * graph->nvtxs);
	cudaMalloc((void**)&rmap,sizeof(int) * graph->nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_memcpy_split, NULL);
	// cudaMemcpy(graph->cuda_xadj,graph->xadj,sizeof(int) * (graph->nvtxs + 1), cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_adjncy,graph->adjncy,sizeof(int) * graph->nedges, cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_adjwgt,graph->adjwgt,sizeof(int) * graph->nedges, cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_vwgt,graph->vwgt,sizeof(int) * graph->nvtxs, cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_where,graph->where,sizeof(int) * graph->nvtxs, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	gettimeofday(&end_memcpy_split, NULL);
	memcpy_split += (end_memcpy_split.tv_sec - begin_memcpy_split.tv_sec) * 1000 + (end_memcpy_split.tv_usec - begin_memcpy_split.tv_usec) / 1000.0;

	// rgraph:where=0, rgraph:where=1
	// 计算左右子图各自的顶点数
	rnvtxs = thrust::reduce(thrust::device, graph->cuda_where, graph->cuda_where + nvtxs);
	lnvtxs = nvtxs - rnvtxs;

	// lgraph
	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&ltxadj,sizeof(int) * (nvtxs + 1));
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	compute_lnedges<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_where,ltxadj);

	thrust::exclusive_scan(thrust::device, ltxadj, ltxadj + nvtxs + 1, ltxadj);
	cudaMemcpy(&lnedges, &ltxadj[nvtxs],sizeof(int), cudaMemcpyDeviceToHost);

	// printf("lgraph ltxadj\n");

	compute_lmap<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_where,lmap);
	thrust::inclusive_scan(thrust::device, lmap, lmap + nvtxs, lmap);

	// printf("lgraph map\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&lgraph->cuda_xadj,sizeof(int) * (lnvtxs + 1));
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	set_lxadj<<<(nvtxs + 128) / 128, 128>>>(nvtxs,lnvtxs,lnedges,graph->cuda_xadj,graph->cuda_where,lmap,ltxadj,lgraph->cuda_xadj);

	// printf("lgraph xadj\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&lgraph->cuda_adjncy,sizeof(int) * lnedges);
	cudaMalloc((void**)&lgraph->cuda_adjwgt,sizeof(int) * lnedges);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	set_ladjncy_ladjwgt<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
		graph->cuda_where,lgraph->cuda_xadj,lgraph->cuda_adjncy,lgraph->cuda_adjwgt,lmap);
	
	// printf("lgraph adjncy\n");

	// vwgt
	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&lgraph->cuda_vwgt,sizeof(int) * lnvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	set_lvwgt<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_where,graph->cuda_vwgt,lmap,lgraph->cuda_vwgt);

	// printf("lgraph vwgt\n");

	// label
	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&lgraph->cuda_label,sizeof(int) * lnvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	if(flag == 1) set_llabel0<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_where,lmap,lgraph->cuda_label);
	else 
	{
		cudaMalloc((void**)&graph->cuda_label,sizeof(int) * graph->nvtxs);
		cudaMemcpy(graph->cuda_label,graph->label,sizeof(int) * graph->nvtxs, cudaMemcpyHostToDevice);
		set_llabel1<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_where,lmap,graph->cuda_label,lgraph->cuda_label);
	}
	// printf("lgraph end\n");

	/*printf("GPU_lgraph:\n");
	cudaDeviceSynchronize();
	exam_graph<<<1,1>>>(lnvtxs,lnedges,lgraph->cuda_xadj,lgraph->cuda_adjncy,lgraph->cuda_vwgt,lgraph->cuda_label);
	cudaDeviceSynchronize();*/

	// rgraph
	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&rtxadj,sizeof(int) * (nvtxs + 1));
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	compute_rnedges<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_where,rtxadj);

	thrust::exclusive_scan(thrust::device, rtxadj, rtxadj + nvtxs + 1, rtxadj);
	cudaMemcpy(&rnedges, &rtxadj[nvtxs],sizeof(int), cudaMemcpyDeviceToHost);

	// printf("rgraph rtxadj\n");

	compute_rmap<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_where,rmap);
	thrust::inclusive_scan(thrust::device, rmap, rmap + nvtxs, rmap);

	// printf("rgraph map\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&rgraph->cuda_xadj,sizeof(int) * (rnvtxs + 1));
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	set_rxadj<<<(nvtxs + 128) / 128, 128>>>(nvtxs,rnvtxs,rnedges,graph->cuda_xadj,graph->cuda_where,rmap,rtxadj,rgraph->cuda_xadj);

	// printf("rgraph xadj\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&rgraph->cuda_adjncy,sizeof(int) * rnedges);
	cudaMalloc((void**)&rgraph->cuda_adjwgt,sizeof(int) * rnedges);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	set_radjncy_radjwgt<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
		graph->cuda_where,rgraph->cuda_xadj,rgraph->cuda_adjncy,rgraph->cuda_adjwgt,rmap);
	
	// printf("rgraph adjncy\n");
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&rgraph->cuda_vwgt,sizeof(int) * rnvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	set_rvwgt<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_where,graph->cuda_vwgt,rmap,rgraph->cuda_vwgt);

	// printf("rgraph vwgt\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_split, NULL);
	cudaMalloc((void**)&rgraph->cuda_label,sizeof(int) * rnvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_split, NULL);
	malloc_split += (end_malloc_split.tv_sec - begin_malloc_split.tv_sec) * 1000 + (end_malloc_split.tv_usec - begin_malloc_split.tv_usec) / 1000.0;
	if(flag == 1) set_rlabel0<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_where,rmap,rgraph->cuda_label);
	else set_rlabel1<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_where,rmap,graph->cuda_label,rgraph->cuda_label);

	// printf("rgraph end\n");

  	set_subgraph_tvwgt(graph, lgraph, rgraph);

	/*printf("GPU_rgraph:\n");
	cudaDeviceSynchronize();
	exam_graph<<<1,1>>>(rnvtxs,rnedges,rgraph->cuda_xadj,rgraph->cuda_adjncy,rgraph->cuda_vwgt,rgraph->cuda_label);
	cudaDeviceSynchronize();*/
  	// printf("lgraph->tvwgt=%d lgraph->tvwgt_reverse=%f rgraph->tvwgt=%d rgraph->tvwgt_reverse=%f\n",lgraph->tvwgt[0],lgraph->tvwgt_reverse[0],rgraph->tvwgt[0],rgraph->tvwgt_reverse[0]);
	
	lgraph->xadj = (int *)malloc(sizeof(int) * (lnvtxs + 1));
	lgraph->adjncy = (int *)malloc(sizeof(int) * lnedges);
	lgraph->adjwgt = (int *)malloc(sizeof(int) * lnedges);
	lgraph->vwgt = (int *)malloc(sizeof(int) * lnvtxs);
	lgraph->label = (int *)malloc(sizeof(int) * lnvtxs);

	rgraph->xadj = (int *)malloc(sizeof(int) * (rnvtxs + 1));
	rgraph->adjncy = (int *)malloc(sizeof(int) * rnedges);
	rgraph->adjwgt = (int *)malloc(sizeof(int) * rnedges);
	rgraph->vwgt = (int *)malloc(sizeof(int) * rnvtxs);
	rgraph->label = (int *)malloc(sizeof(int) * rnvtxs);

	cudaDeviceSynchronize();
	gettimeofday(&begin_memcpy_split, NULL);
	cudaMemcpy(lgraph->xadj,lgraph->cuda_xadj,sizeof(int) * (lnvtxs + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(lgraph->adjncy,lgraph->cuda_adjncy,sizeof(int) * lnedges, cudaMemcpyDeviceToHost);
	cudaMemcpy(lgraph->adjwgt,lgraph->cuda_adjwgt,sizeof(int) * lnedges, cudaMemcpyDeviceToHost);
	cudaMemcpy(lgraph->vwgt,lgraph->cuda_vwgt,sizeof(int) * lnvtxs, cudaMemcpyDeviceToHost);
	cudaMemcpy(lgraph->label,lgraph->cuda_label,sizeof(int) * lnvtxs, cudaMemcpyDeviceToHost);

	cudaMemcpy(rgraph->xadj,rgraph->cuda_xadj,sizeof(int) * (rnvtxs + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(rgraph->adjncy,rgraph->cuda_adjncy,sizeof(int) * rnedges, cudaMemcpyDeviceToHost);
	cudaMemcpy(rgraph->adjwgt,rgraph->cuda_adjwgt,sizeof(int) * rnedges, cudaMemcpyDeviceToHost);
	cudaMemcpy(rgraph->vwgt,rgraph->cuda_vwgt,sizeof(int) * rnvtxs, cudaMemcpyDeviceToHost);
	cudaMemcpy(rgraph->label,rgraph->cuda_label,sizeof(int) * rnvtxs, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	gettimeofday(&end_memcpy_split, NULL);
	memcpy_split += (end_memcpy_split.tv_sec - begin_memcpy_split.tv_sec) * 1000 + (end_memcpy_split.tv_usec - begin_memcpy_split.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_free_split, NULL);
	cudaFree(lmap);
	cudaFree(rmap);
	cudaFree(ltxadj);
	cudaFree(rtxadj);

	cudaFree(graph->cuda_xadj);
	cudaFree(graph->cuda_adjncy);
	cudaFree(graph->cuda_adjwgt);
	cudaFree(graph->cuda_vwgt);
	cudaFree(graph->cuda_label);
	cudaFree(graph->cuda_where);

	cudaFree(lgraph->cuda_xadj);
	cudaFree(lgraph->cuda_adjncy);
	cudaFree(lgraph->cuda_adjwgt);
	cudaFree(lgraph->cuda_label);
	cudaFree(lgraph->cuda_xadj);

	cudaFree(lgraph->cuda_xadj);
	cudaFree(lgraph->cuda_adjncy);
	cudaFree(lgraph->cuda_adjwgt);
	cudaFree(lgraph->cuda_vwgt);
	cudaFree(lgraph->cuda_label);
	cudaDeviceSynchronize();
	gettimeofday(&end_free_split, NULL);
	free_split += (end_free_split.tv_sec - begin_free_split.tv_sec) * 1000 + (end_free_split.tv_usec - begin_free_split.tv_usec) / 1000.0;

	lgraph->nvtxs  = lnvtxs;
	lgraph->nedges = lnedges;
	rgraph->nvtxs  = rnvtxs;
	rgraph->nedges = rnedges;

	*r_lgraph=lgraph;
	*r_rgraph=rgraph;
}

#endif