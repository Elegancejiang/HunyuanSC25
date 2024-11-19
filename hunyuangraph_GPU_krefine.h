#ifndef _H_KREFINE
#define _H_KREFINE

#include "hunyuangraph_struct.h"

/*CUDA-get the max/min pwgts*/
__global__ void Sum_maxmin_pwgts(int *maxwgt, int *minwgt, float *tpwgts, int tvwgt, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nparts)
	{
		float result = tpwgts[ii] * tvwgt;

		maxwgt[ii] = int(result * 1.03);
		minwgt[ii] = int(result / 1.03);
	}
}

/*CUDA-init boundary vertex num*/
__global__ void initbndnum(int *bndnum)
{
  bndnum[0] = 0;
}

/*CUDA-find vertex where ed-id>0 */
__global__ void Find_real_bnd_info(int *cuda_real_bnd_num, int *cuda_real_bnd, int *where, \
	int *xadj, int *adjncy, int *adjwgt, int nvtxs, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs) // && moved[ii] == 0
	{
		int i, k, begin, end, me, other, from;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		me    = 0;
		other = 0;
		from  = where[ii];

		for(i = begin;i < end;i++)
		{
			k = adjncy[i];
			if(where[k] == from) me += adjwgt[i];
			else other += adjwgt[i];
		}
		if(other > me) cuda_real_bnd[atomicAdd(&cuda_real_bnd_num[0],1)] = ii;
	}
}

/**/
__global__ void init_bnd_info(int *bnd_info, int length)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < length)
    bnd_info[ii] = 0;
}

/*CUDA-find boundary vertex should ro which part*/
__global__ void find_kayparams(int *cuda_real_bnd_num, int *bnd_info, int *cuda_real_bnd, int *where, \
int *xadj, int *adjncy, int *adjwgt, int nparts, int *cuda_bn, int *cuda_bt, int *cuda_g)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<cuda_real_bnd_num[0])
  {
    int pi, other, i, k, me_wgt, other_wgt;
    int start, end, begin, last;

    pi    = cuda_real_bnd[ii];
    begin = xadj[pi];
    last  = xadj[pi+1];
    start = nparts * ii;
    end   = nparts * (ii+1);
    other = where[pi];

    for(i = begin;i < last;i++)
    {
      k = adjncy[i];
      k = start + where[k];
      bnd_info[k] += adjwgt[i];
    }

    me_wgt = other_wgt = bnd_info[start + other];

    for(i=start;i<end;i++)
    {
      k = bnd_info[i];
      if(k > other_wgt)
      {
        other_wgt = k;
        other     = i - start;
      }
    }

    cuda_g[ii]  = other_wgt - me_wgt;
    cuda_bt[ii] = other;
    cuda_bn[ii] = pi;

  }
}

/*CUDA-init params*/
__global__ void initcucsr(int *cu_csr, int *bndnum)
{
  cu_csr[0] = 0;
  cu_csr[1] = bndnum[0];
}

/**/
__global__ void init_cu_que(int *cuda_que, int length)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < length)
    cuda_que[ii] = -1;
}

/*CUDA-get a csr array*/
__global__ void findcsr(int *cuda_bt, int *cuda_que, int *bnd_num, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < nparts)
  {
    int i, t;
    int begin, end;
    begin = 2 * ii;
    end   = bnd_num[0];

    for(i = 0;i < end;i++)
    {
      if(ii == cuda_bt[i])
      {
        cuda_que[begin] = i;
        break; 
      }
    }

    t = cuda_que[begin];

    if(t!=-1)
    {
      for(i = t;i < end;i++)
      {
        if(cuda_bt[i] != ii)
        {
          cuda_que[begin + 1] = i - 1;
          break; 
        }
      }
    }

    t = 2 * cuda_bt[end - 1] + 1;
    cuda_que[t] = end - 1;
  }
}

/*Find boundary vertex information*/
void hunyuangraph_findgraphbndinfo(hunyuangraph_admin_t *hunyuangraph_admin,hunyuangraph_graph_t *graph)
{
	int nvtxs  = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;
	int bnd_num; 

	initbndnum<<<1,1>>>(graph->cuda_bndnum);

	Find_real_bnd_info<<<nvtxs / 32 + 1,32>>>(graph->cuda_bndnum,graph->cuda_bnd,graph->cuda_where,\
		graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,nvtxs,nparts);
  
	cudaMemcpy(&bnd_num,graph->cuda_bndnum, sizeof(int), cudaMemcpyDeviceToHost);
  
	if(bnd_num > 0)
	{
		// cudaMalloc((void**)&graph->cuda_info, bnd_num * nparts * sizeof(int));
		graph->cuda_info = (int *)rmalloc_with_check(sizeof(int) * bnd_num * nparts,"info");


		init_bnd_info<<<bnd_num * nparts / 32 + 1,32>>>(graph->cuda_info, bnd_num * nparts);

		find_kayparams<<<bnd_num/32+1,32>>>(graph->cuda_bndnum,graph->cuda_info,graph->cuda_bnd,graph->cuda_where,\
			graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,nparts,graph->cuda_bn,graph->cuda_bt,graph->cuda_g);

		initcucsr<<<1,1>>>(graph->cuda_csr,graph->cuda_bndnum);

		bb_segsort(graph->cuda_bt, graph->cuda_bn, bnd_num, graph->cuda_csr, 1);
		
		init_cu_que<<<2 * nparts / 32 + 1,32>>>(graph->cuda_que, 2 * nparts);
		
		findcsr<<<nparts/32+1,32>>>(graph->cuda_bt,graph->cuda_que,graph->cuda_bndnum,nparts);
		
		rfree_with_check(sizeof(int) * bnd_num * nparts,"info");
		// cudaFree(graph->cuda_info);
	}

	graph->cpu_bndnum=(int *)malloc(sizeof(int));
	graph->cpu_bndnum[0]=bnd_num;
}

/*CUDA-move vertex*/
__global__ void Exnode_part1(int *cuda_que, int *pwgts, int *bnd, int *bndto, int *vwgt,\
  int *maxvwgt, int *minvwgt, int *where, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<nparts)
  {
    int i, me, to, vvwgt;
    int memax, memin, tomax, tomin, mepwgts, topwgts;
    int begin, end, k;
    begin = cuda_que[2 * ii];
    end   = cuda_que[2 * ii + 1];
    if(begin != -1)
    {
      for(i = begin;i <= end;i++)
      {
        k     = bnd[i];
        vvwgt = vwgt[k];
        me    = where[k];
        to    = bndto[i];

        memax   = maxvwgt[me];
        memin   = minvwgt[me];
        tomax   = maxvwgt[to];
        tomin   = minvwgt[to];
        mepwgts = pwgts[me];
        topwgts = pwgts[to];

        if(me <= to)
        {
          if(((topwgts + vvwgt >= tomin) && (topwgts + vvwgt <= tomax))\
            &&((mepwgts - vvwgt >= memin) && (mepwgts - vvwgt <= memax)))
          {
            atomicAdd(&pwgts[to],vvwgt);
            atomicSub(&pwgts[me],vvwgt);
            where[k] = to;
          }
        }
      }
    }
  }
}

/*CUDA-move vertex*/
__global__ void Exnode_part2(int *cuda_que, int *pwgts, int *bnd, int *bndto, int *vwgt,\
  int *maxvwgt, int *minvwgt, int *where, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<nparts)
  {
    int i, me, to, vvwgt;
    int memax, memin, tomax, tomin, mepwgts, topwgts;
    int begin, end, k;
    begin = cuda_que[2 * ii];
    end   = cuda_que[2 * ii + 1];
    if(begin != -1)
    {
      for(i = begin;i <= end;i++)
      {
        k     = bnd[i];
        vvwgt = vwgt[k];
        me    = where[k];
        to    = bndto[i];

        memax   = maxvwgt[me];
        memin   = minvwgt[me];
        tomax   = maxvwgt[to];
        tomin   = minvwgt[to];
        mepwgts = pwgts[me];
        topwgts = pwgts[to];

        if(me > to)
        {
          if(((topwgts+vvwgt>=tomin)&&(topwgts+vvwgt<=tomax))\
          &&((mepwgts-vvwgt>=memin)&&(mepwgts-vvwgt<=memax)))
          {
            atomicAdd(&pwgts[to],vvwgt);
            atomicSub(&pwgts[me],vvwgt);
            where[k] = to;
          }
        }
      }
    }
  }
}

/*Graph multilevel uncoarsening algorithm*/
void hunyuangraph_k_refinement(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs  = graph->nvtxs;

	Sum_maxmin_pwgts<<<nparts / 32 + 1,32>>>(graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_tpwgts,graph->tvwgt[0],nparts);

	for(int i = 0;i < 2;i++)
	{
		hunyuangraph_findgraphbndinfo(hunyuangraph_admin,graph);

		if(graph->cpu_bndnum[0] > 0)
		{
			Exnode_part1<<<nparts/32+1,32>>>(graph->cuda_que,graph->cuda_pwgts,graph->cuda_bn,graph->cuda_bt,graph->cuda_vwgt,\
				graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_where,nparts);

			Exnode_part2<<<nparts/32+1,32>>>(graph->cuda_que,graph->cuda_pwgts,graph->cuda_bn,graph->cuda_bt,graph->cuda_vwgt,\
				graph->cuda_maxwgt,graph->cuda_minwgt,graph->cuda_where,nparts);
		}
		else
			break;
	}
}

#endif