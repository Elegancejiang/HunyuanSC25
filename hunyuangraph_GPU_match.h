#ifndef _H_GPU_MATCH
#define _H_GPU_MATCH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_GPU_prefixsum.h"

/*CUDA-init match array*/
__global__ void init_gpu_match(int *match, int nvtxs)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < nvtxs)
        match[ii] = -1;
}

/*CUDA-hem matching*/
__global__ void cuda_hem(int nvtxs_hem, int *match, int *xadj, int *vwgt,int *adjwgt, int *adjncy, int maxvwgt_hem)
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

	tt      = 1024;
	nvtxs   = nvtxs_hem;
	maxvwgt = maxvwgt_hem;

	if(nvtxs % tt == 0)
	{
		b = nvtxs / tt;
		ibegin = ii * b;
		iend   = ibegin + b;
	}
	else 
	{
		b = nvtxs / tt;
		a = b + 1;
		x = nvtxs - b * tt;
		if(ii < x)
		{
			ibegin = ii * a;
			iend   = ibegin + a;
		}
		else
		{
			ibegin = ii * b + x;
			iend   = ibegin + b;
		}
	}
	for(i = ibegin;i < iend;i++)
	{
		if(match[i] == -1)
		{
			begin = xadj[i];
			end   = xadj[i + 1];
			ivwgt = vwgt[i];

			maxidx = i;
			maxwgt = -1;

			if (ivwgt < maxvwgt) 
			{
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
			}

			if(maxidx != -1)
			{
				atomicCAS(&match[maxidx],-1,i);
				atomicExch(&match[i],maxidx);
			}
		}
	}
}

__global__ void cuda_hem_test(int nvtxs_hem, int *match, int *xadj, int *vwgt,int *adjwgt, int *adjncy, int maxvwgt_hem)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < nvtxs_hem)
	{
		if(ii % 2 == 0) match[ii] = ii + 1;
		else match[ii] = ii - 1;

		if(ii == nvtxs_hem - 1 && ii % 2 == 0) match[ii] = ii;
	}
}

__global__ void cuda_hem_229_3(int nvtxs, int *match, int *xadj, int *vwgt,int *adjwgt, int *adjncy, int maxvwgt)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int maxidx, maxwgt, j, k, ivwgt, jw;
		int begin, end;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		ivwgt = vwgt[ii];

		if(match[ii] == -1)
		{
			begin = xadj[ii];
			end   = xadj[ii + 1];
			ivwgt = vwgt[ii];

			maxidx = ii;
			maxwgt = -1;

			if (ivwgt < maxvwgt) 
			{
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
				if(maxidx == ii && 3 * ivwgt < maxvwgt)
					maxidx = -1;
			}

			if(maxidx != -1)
			{
				atomicCAS(&match[maxidx],-1,ii);
				atomicExch(&match[ii],maxidx);
			}
		}
	}
}

__global__ void reset_match(int nvtxs, int *match)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int t = atomicAdd(&match[ii],0);
		if(t != -1)
		{
			if(match[t] != ii) 
				atomicExch(&match[ii],-1);
		}
	}
}

/*CUDA-set conflict array*//*cuda_cleanv*/
/*CUDA-find cgraph vertex part1-remark the match array by s*//*findc1*/
/*CUDA-find cgraph vertex part2-make sure the pair small label vertex*//*findc2*/
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

/*CUDA-find cgraph vertex part4-make sure vertex pair real rdge*//*findc4*/
__global__ void resolve_conflict_4(int *match, int *cmap, int *txadj, int *xadj, int *cvwgt, int *vwgt, int nvtxs)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < nvtxs)
    {
        int u = match[ii];
        if(ii > u)
        {
            int t = cmap[u];
            cmap[ii] = t;
            cvwgt[t] = vwgt[ii] + vwgt[u];
        }
        else 
        {
            int t, begin, end, length; 
            t = cmap[ii];
            begin  = xadj[ii];
            end    = xadj[ii + 1];
            length = end - begin;
            if(u != ii)
            {
                begin = xadj[u];
                end   = xadj[u + 1];
                txadj[t + 1] = length + end - begin;
            }
            else 
				txadj[t + 1] = length;

            if(ii == u) cvwgt[t] = vwgt[ii];
        }
		if(ii == 0)
			txadj[0] = 0;
    }
}

/*Get gpu graph matching params by hem*/
hunyuangraph_graph_t *hunyuangraph_gpu_match(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
    int nvtxs  = graph->nvtxs;
    int nedges = graph->nedges;
    int cnvtxs = 0;
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match,NULL);
	init_gpu_match<<<(nvtxs + 127) / 128,128>>>(graph->cuda_match,nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match,NULL);
	init_gpu_match_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match,NULL);
	
	// end_version
	for(int i = 0;i < 1;i++)
	{
		cuda_hem_229_3<<<(nvtxs + 127) / 128,128>>>(nvtxs,graph->cuda_match,graph->cuda_xadj,graph->cuda_vwgt,graph->cuda_adjwgt,graph->cuda_adjncy,\
			hunyuangraph_admin->maxvwgt);
		cudaDeviceSynchronize();
		
		reset_match<<<(nvtxs + 127) / 128,128>>>(nvtxs,graph->cuda_match);
		cudaDeviceSynchronize();
		
		cuda_hem<<<1024,1>>>(nvtxs,graph->cuda_match,graph->cuda_xadj,graph->cuda_vwgt,graph->cuda_adjwgt,graph->cuda_adjncy,\
        	hunyuangraph_admin->maxvwgt);
	}
	
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match,NULL);
	hem_gpu_match_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match,NULL);
    resolve_conflict_1<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match,NULL);
	resolve_conflict_1_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match,NULL);
    resolve_conflict_2<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_match, graph->cuda_cmap, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match,NULL);
	resolve_conflict_2_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match,NULL);
    // thrust::inclusive_scan(thrust::device, graph->cuda_cmap, graph->cuda_cmap + nvtxs, graph->cuda_cmap);
	prefixsum(graph->cuda_cmap,graph->cuda_cmap,nvtxs,prefixsum_blocksize,0);	//0:lmalloc,1:rmalloc
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match,NULL);
	inclusive_scan_time1 += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;
    
	cudaMemcpy(&cnvtxs,&graph->cuda_cmap[nvtxs - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cnvtxs++;

    hunyuangraph_graph_t *cgraph = hunyuangraph_set_gpu_cgraph(graph, cnvtxs); 
    cgraph->nvtxs = cnvtxs;

	cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
    // cudaMalloc((void**)&graph->txadj, (cnvtxs + 1) * sizeof(int));
	graph->txadj = (int *)rmalloc_with_check(sizeof(int) * (cnvtxs + 1),"txadj");
    // cudaMalloc((void**)&cgraph->cuda_vwgt, cnvtxs * sizeof(int));
	cgraph->cuda_vwgt = (int *)lmalloc_with_check(sizeof(int) * cnvtxs,"vwgt");
	cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_match,NULL);
    resolve_conflict_4<<<(nvtxs + 127) / 128,128>>>(graph->cuda_match, graph->cuda_cmap, graph->txadj, graph->cuda_xadj, \
        cgraph->cuda_vwgt, graph->cuda_vwgt, nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_match,NULL);
	resolve_conflict_4_time += (end_gpu_match.tv_sec - begin_gpu_match.tv_sec) * 1000 + (end_gpu_match.tv_usec - begin_gpu_match.tv_usec) / 1000.0;
    
    return cgraph;
}

#endif