#ifndef _H_GPU_CONSTRUCTION
#define _H_GPU_CONSTRUCTION

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_GPU_prefixsum.h"
#include "bb_segsort.h"

/*CUDA-set each vertex pair adjacency list and weight params*/
__global__ void set_tadjncy_tadjwgt(int *txadj, int *xadj, int *match, int *adjncy,\
    int *cmap, int *tadjncy, int *tadjwgt, int *adjwgt, int nvtxs)
{
    // long long int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int ii  = blockIdx.x * 4 + threadIdx.x / 32;
    int tid = threadIdx.x % 32;

    if(ii < nvtxs)
    {
        int u, t, pp, k, ptr, begin, end, iii;

        u = match[ii];
        t = cmap[ii];

        pp = txadj[t];
        if(ii > u)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            pp   += end - begin;
        }

        begin = xadj[ii];
        end   = xadj[ii + 1];

        for(iii = begin + tid, ptr = pp + tid;iii < end;iii += 32, ptr += 32)
        {
            k   = adjncy[iii];

            tadjncy[ptr] = cmap[k];
            tadjwgt[ptr] = adjwgt[iii];
        }
    }
}

__global__ void segment_sort(int *tadjncy, int *tadjwgt, int nedges, int *txadj, int cnvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < cnvtxs)
	{
		int begin, end, ptr, val;
		int i, j, k;
		begin = txadj[ii];
		end   = txadj[ii + 1];

		for(i = begin;i < end;i++)
		{
			ptr = i;
			val = tadjwgt[ptr];
			for(j = i + 1;j < end;j++)
				if(tadjwgt[j] < val) ptr = j, val = tadjwgt[ptr];
			val = tadjncy[ptr], tadjncy[ptr] = tadjncy[i], tadjncy[i] = val;
			val = tadjwgt[ptr], tadjwgt[ptr] = tadjwgt[i], tadjwgt[i] = val;
		}
	}
}

//Sort_cnedges2_part1<<<(cnvtxs + 3) / 4,128>>>(graph->tadjncy,graph->txadj,temp_scan,cnvtxs);
/*CUDA-Segmentation sorting part1-set scan array value 0 or 1*/
__global__ void mark_edges(int *tadjncy, int *txadj, int *temp_scan, int cnvtxs)
{
    // long long int i   = blockIdx.x * blockDim.x + threadIdx.x;
	int ii  = blockIdx.x * 4 + threadIdx.x / 32;
	int tid = threadIdx.x % 32;

	if(ii < cnvtxs)
	{
		int j, begin, end, iii;

		begin  = txadj[ii];
		end    = txadj[ii + 1];

		for(iii = begin + tid;iii < end;iii += 32)
		{
			j   = tadjncy[iii];
				
			if(iii == begin)
			{
				if(j == ii) temp_scan[iii] = 0;
				else temp_scan[iii] = 1;
			}
			else 
			{
				if(j == ii) temp_scan[iii] = 0;
				else
				{
					if(j == tadjncy[iii - 1]) temp_scan[iii] = 0;
					else temp_scan[iii] = 1;
				}
			}
		}
	}
}

__global__ void Sort_cnedges2_part1_shared(int *tadjncy, int *txadj, int *temp_scan, int cnvtxs)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int ii  = i / 32;
    int tid = i - ii * 32;

    __shared__ int cache_txadj[8];
    // __shared__ int cache_tadjncy[4][32];
    int pid = ii % 4;

    if(ii < cnvtxs)
    {
        int j, begin, end;

        if(tid == 0) cache_txadj[pid] = txadj[ii];
        if(tid == 1) cache_txadj[pid + 1] = txadj[ii + 1];

        __syncthreads();

        // begin  = txadj[ii];
        // end    = txadj[ii + 1];

        //bank conflict
        begin = cache_txadj[pid];
        end   = cache_txadj[pid + 1];

        for(i = begin + tid;i < end;i += 32)
        {
            j   = tadjncy[i];
            // cache_tadjncy[pid][tid] = j;
            // __syncthreads();
            
            if(i == begin)
            {
                if(j == ii) temp_scan[i] = 0;
                else temp_scan[i] = 1;
            }
            else 
            {
                if(j == ii) temp_scan[i] = 0;
                else
                {
                    if(j == tadjncy[i - 1]) temp_scan[i] = 0;
                    // if(tid != 0 && j == cache_tadjncy[pid][tid - 1]) temp_scan[i] = 0;
                    // else if(tid == 0 && j == tadjncy[i - 1]) temp_scan[i] = 0;
                    else temp_scan[i] = 1;
                }
            }
        }
    }
}

/*CUDA-Segmentation sorting part2-set cxadj*/
__global__ void set_cxadj(int *txadj, int *temp_scan, int *cxadj, int cnvtxs)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < cnvtxs)
    { 
        int ppp = txadj[ii + 1];

        // cxadj[ii + 1] = temp_scan[ppp - 1];
        if(ppp > 0)
            cxadj[ii + 1] = temp_scan[ppp - 1];
        else 
            cxadj[ii + 1] = 0;
    }
    else if(ii == cnvtxs) cxadj[0] = 0;
} 

/*CUDA-Segmentation sorting part2.5-init cadjwgt and cadjncy*/
__global__ void init_cadjwgt(int *cadjwgt, int cnedges)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;  

    if(ii < cnedges)
        cadjwgt[ii] = 0;
}

/*CUDA-Segmentation sorting part3-deduplication and accumulation*/
__global__ void set_cadjncy_cadjwgt(int *tadjncy,int *txadj, int *tadjwgt,int *temp_scan, int *cxadj,int *cadjncy, int *cadjwgt, int cnvtxs)
{
    // long long int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int ii  = blockIdx.x * 4 + threadIdx.x / 32;
    int tid = threadIdx.x % 32;

    // if(ii < cnvtxs)
    // {
    //     int ptr, j, begin, end;

    //     begin = txadj[ii];
    //     end   = txadj[ii + 1];

    //     for(i = begin + tid;i < end;i += 32)
    //     {
    //         j   = tadjncy[i];
    //         ptr = temp_scan[i] - 1;

    //         if(j != ii)
    //         {
    //             atomicAdd(&cadjwgt[ptr],tadjwgt[i]);

    //             if(i == begin) cadjncy[ptr] = j;
    //             else
    //             {
    //                 if(j != tadjncy[i - 1]) cadjncy[ptr] = j;
    //             }
    //         }
    //     }
    // }

	if(ii < cnvtxs)
	{
		int begin, end, j, k, iii;

		begin = txadj[ii];
        end   = txadj[ii + 1];

		for(iii = begin + tid;iii < end;iii += 32)
		{
			j = tadjncy[iii];
			k = temp_scan[iii] - 1;

			if(iii == begin)
			{
				if(j != ii)
				{
					cadjncy[k] = j;
					atomicAdd(&cadjwgt[k],tadjwgt[iii]);
				}
			}
			else 
			{
				if(j != ii)
				{
					if(j != tadjncy[iii - 1])
					{
						cadjncy[k] = j;
						atomicAdd(&cadjwgt[k],tadjwgt[iii]);
					}
					else atomicAdd(&cadjwgt[k],tadjwgt[iii]);
				}
			}
		}
	}
}

__global__ void print_xadj(int nvtxs, int *xadj)
{
	int i;
	printf("xadj:");
	for(i = 0;i < nvtxs;i++)
		printf("%10d ", xadj[i]);
	printf("\n");
}

/*Create gpu coarsen graph by contract*/
void hunyuangraph_gpu_create_cgraph(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, hunyuangraph_graph_t *cgraph)
{
    int nvtxs  = graph->nvtxs;
    int nedges = graph->nedges;
    int cnvtxs = cgraph->nvtxs;

    // printf("txadj\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(11, graph->txadj);
	// cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
    if(GPU_Memory_Pool)
    {
        prefixsum(graph->txadj + 1, graph->txadj + 1, cnvtxs, prefixsum_blocksize, 1);	//0:lmalloc,1:rmalloc
    }
    else
    {
        thrust::inclusive_scan(thrust::device, graph->txadj, graph->txadj + cnvtxs + 1, graph->txadj);
        // thrust::exclusive_scan(thrust::device, graph->txadj, graph->txadj + cnvtxs + 1, graph->txadj);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    exclusive_scan_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(11, graph->txadj);
	// cudaDeviceSynchronize();

    // printf("prefixsum end\n");
    // cudaMalloc((void**)&graph->tadjncy,nedges * sizeof(int));
    // cudaMalloc((void**)&graph->tadjwgt,nedges * sizeof(int));
	int *bb_keysB_d, *bb_valsB_d;
    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
    if(GPU_Memory_Pool)
    {
        bb_keysB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"bb_keysB_d");
        bb_valsB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"bb_valsB_d");
        graph->tadjncy = (int *)rmalloc_with_check(sizeof(int) * nedges,"tadjncy");
        graph->tadjwgt = (int *)rmalloc_with_check(sizeof(int) * nedges,"tadjwgt");
    }
    else
    {
        cudaMalloc((void**)&graph->tadjncy, sizeof(int) * nedges);
        cudaMalloc((void**)&graph->tadjwgt, sizeof(int) * nedges);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
    set_tadjncy_tadjwgt<<<(nvtxs + 3) / 4,128>>>(graph->txadj,graph->cuda_xadj,graph->cuda_match,graph->cuda_adjncy,graph->cuda_cmap,\
        graph->tadjncy,graph->tadjwgt,graph->cuda_adjwgt,nvtxs);
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    set_tadjncy_tadjwgt_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
	// printf("set_tadjncy_tadjwgt end\n");
    // printf("tadjncy/tadjwgt\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(160, graph->tadjncy);
	// cudaDeviceSynchronize();
    // print_xadj<<<1, 1>>>(160, graph->tadjwgt);
	// cudaDeviceSynchronize();

    if(GPU_Memory_Pool)
    {
        int *bb_counter, *bb_id;
        cudaDeviceSynchronize();
        gettimeofday(&begin_malloc,NULL);
        // bb_keysB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"bb_keysB_d");
        // bb_valsB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"bb_valsB_d");
        bb_id      = (int *)rmalloc_with_check(sizeof(int) * cnvtxs,"bb_id");
        bb_counter = (int *)rmalloc_with_check(sizeof(int) * 13,"bb_counter");
        cudaDeviceSynchronize();
        gettimeofday(&end_malloc,NULL);
        coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
        // printf("hunyuangraph_segmengtsort malloc end\n");
        cudaDeviceSynchronize();
        gettimeofday(&begin_gpu_contraction,NULL);
        hunyuangraph_segmengtsort(graph->tadjncy, graph->tadjwgt, nedges, graph->txadj, cnvtxs, bb_counter, bb_id, bb_keysB_d, bb_valsB_d);
        // segment_sort<<<(cnvtxs + 127) / 128, 128>>>(graph->tadjncy, graph->tadjwgt, nedges, graph->txadj, cnvtxs);
        cudaDeviceSynchronize();
        gettimeofday(&end_gpu_contraction,NULL);
        ncy_segmentsort_gpu_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
        
        // printf("hunyuangraph_segmengtsort end\n");
        cudaDeviceSynchronize();
        gettimeofday(&begin_free,NULL);
        rfree_with_check((void *)bb_counter, sizeof(int) * 13,"bb_counter");		//bb_counter
        rfree_with_check((void *)bb_id, sizeof(int) * cnvtxs,"bb_id");				//bb_id
        graph->tadjncy = bb_keysB_d;
        graph->tadjwgt = bb_valsB_d;
        rfree_with_check((void *)graph->tadjwgt, sizeof(int) * nedges,"bb_valsB_d");		//tadjwgt
        rfree_with_check((void *)graph->tadjncy, sizeof(int) * nedges,"bb_keysB_d");		//tadjncy
        cudaDeviceSynchronize();
        gettimeofday(&end_free,NULL);
        coarsen_free += (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
    }
    else
    {
        cudaDeviceSynchronize();
        gettimeofday(&begin_gpu_contraction,NULL);
        bb_segsort(graph->tadjncy, graph->tadjwgt, nedges, graph->txadj, cnvtxs);
        cudaDeviceSynchronize();
        gettimeofday(&end_gpu_contraction,NULL);
        ncy_segmentsort_gpu_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    }
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(160, graph->tadjncy);
	// cudaDeviceSynchronize();
    // print_xadj<<<1, 1>>>(160, graph->tadjwgt);
	// cudaDeviceSynchronize();

    int *temp_scan;
    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
    // cudaMalloc((void**)&temp_scan, nedges * sizeof(int));
    if(GPU_Memory_Pool)
    	temp_scan = (int *)rmalloc_with_check(sizeof(int) * nedges,"temp_scan");
    else    
        cudaMalloc((void**)&temp_scan, sizeof(int) * nedges);
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
    mark_edges<<<(cnvtxs + 3) / 4,128>>>(graph->tadjncy,graph->txadj,temp_scan,cnvtxs);
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    mark_edges_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    // printf("mark_edges end\n");
    // printf("temp_scan\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(160, temp_scan);
	// cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
    if(GPU_Memory_Pool)
        prefixsum(temp_scan,temp_scan,nedges,prefixsum_blocksize,1);	//0:lmalloc,1:rmalloc
    else
        thrust::inclusive_scan(thrust::device,temp_scan, temp_scan + nedges, temp_scan);
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    inclusive_scan_time2 += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    // printf("prefixsum end\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(160, temp_scan);
	// cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
    // cudaMalloc((void**)&cgraph->cuda_xadj, (cnvtxs+1)*sizeof(int));
    if(GPU_Memory_Pool)
    	cgraph->cuda_xadj = (int *)lmalloc_with_check(sizeof(int) * (cnvtxs + 1),"xadj");
    else 
        cudaMalloc((void**)&cgraph->cuda_xadj, sizeof(int) * (cnvtxs + 1));
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
    set_cxadj<<<(cnvtxs + 128) / 128,128>>>(graph->txadj,temp_scan,cgraph->cuda_xadj,cnvtxs);
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    set_cxadj_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    // printf("set_cxadj end\n");
    // printf("cxadj\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(11, cgraph->cuda_xadj);
	// cudaDeviceSynchronize();

    cudaMemcpy(&cgraph->nedges, &cgraph->cuda_xadj[cnvtxs], sizeof(int), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
    // cudaMalloc((void**)&cgraph->cuda_adjncy, cgraph->nedges * sizeof(int));
    // cudaMalloc((void**)&cgraph->cuda_adjwgt, cgraph->nedges * sizeof(int));
    if(GPU_Memory_Pool)
    {
        cgraph->cuda_adjncy = (int *)lmalloc_with_check(sizeof(int) * cgraph->nedges,"adjncy");
        cgraph->cuda_adjwgt = (int *)lmalloc_with_check(sizeof(int) * cgraph->nedges,"adjwgt");
    }
    else
    {
        cudaMalloc((void**)&cgraph->cuda_adjncy, sizeof(int) * cgraph->nedges);
        cudaMalloc((void**)&cgraph->cuda_adjwgt, sizeof(int) * cgraph->nedges);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
    
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
    init_cadjwgt<<<(cgraph->nedges + 127) / 128,128>>>(cgraph->cuda_adjwgt,cgraph->nedges);
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    init_cadjwgt_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    // printf("init_cadjwgt end\n");
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
    set_cadjncy_cadjwgt<<<(cnvtxs + 3) / 4,128>>>(graph->tadjncy,graph->txadj,\
        graph->tadjwgt,temp_scan,cgraph->cuda_xadj,cgraph->cuda_adjncy,cgraph->cuda_adjwgt,cnvtxs);
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    set_cadjncy_cadjwgt_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(11, cgraph->cuda_xadj);
	// cudaDeviceSynchronize();
	cgraph->tvwgt[0] = graph->tvwgt[0];  

	// printf("cnvtxs=%d cnedges=%d ",cnvtxs,cgraph->nedges);
    
    cudaDeviceSynchronize();
    gettimeofday(&begin_free,NULL);

    if(GPU_Memory_Pool)
    {
        rfree_with_check((void *)temp_scan, sizeof(int) * nedges,"temp_scan");			//temp_scan
        rfree_with_check((void *)graph->tadjwgt, sizeof(int) * nedges,"tadjwgt");		//tadjwgt
        rfree_with_check((void *)graph->tadjncy, sizeof(int) * nedges,"tadjncy");		//tadjncy
        rfree_with_check((void *)graph->txadj, sizeof(int) * (cnvtxs + 1),"txadj");		//txadj
        rfree_with_check((void *)graph->cuda_match, sizeof(int) * nvtxs,"match");		//match
    }
    else
    {
        cudaFree(temp_scan);
        cudaFree(graph->tadjwgt);
        cudaFree(graph->tadjncy);
        cudaFree(graph->txadj);
        cudaFree(graph->cuda_match);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end_free,NULL);
    coarsen_free += (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
}

#endif