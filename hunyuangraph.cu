
#include"HunyuanGRAPH.h"

/*Main function*/
int main(int argc, char **argv)
{  
    cudaSetDevice(0);

    char *filename = (argv[1]);
    int nparts     = atoi(argv[2]);

    hunyuangraph_graph_t *graph = hunyuangraph_readgraph(filename);

    printf("graph:%s %d %d\n",filename,graph->nvtxs,graph->nedges);
  
    int *part = (int*)malloc(sizeof(int) * graph->nvtxs);

    float tpwgts[nparts];
    for(int i = 0;i < nparts;i++)
        tpwgts[i] = 1.0 / nparts;
  
    float ubvec = 1.03;
    
    hunyuangraph_PartitionGraph(&graph->nvtxs, graph->xadj, graph->adjncy, graph->vwgt, graph->adjwgt, &nparts, tpwgts, &ubvec, part);

//   printf("graph:%s %d %d\n",filename,graph->nvtxs,graph->nedges);
	print_time_all(graph, part, hunyuangraph_computecut(graph, part));

	//hunyuangraph_writetofile(filename, part, graph->nvtxs, nparts); 

	//   printf("\n");
	//   printf("the match            %lf\n",part_match+part_cmatch);
	//   printf("the multi-node       %lf\n",part_contract+part_ccontract+part_2map+part_map);
	// printf("the 2refine          %10.2lf ms\n",part_2refine);
	// printf("the 2refine_gpu      %10.2lf ms\n",gpu_2way + malloc_2way);
	//   printf("the krefine          %lf\n",part_krefine);
	// printf("bfs=                 %10.2lf ms\n",part_bfs);
	// printf("slipt=               %10.2lf ms\n",part_slipt);
	// printf("2refine_gpu time=    %d\n",n);
	//   printf("\n\n\n\n");

	// printf("\n");
	// printf("the match            %10.2lf ms\n",part_match);
	// printf("the contract         %10.2lf ms\n",part_contract);
	// printf("the cmatch           %lf\n",part_cmatch);
	// printf("the ccontract        %lf\n",part_ccontract);
	// printf("the GGP(bfs)         %lf\n",part_bfs);
	// printf("the 2refine          %lf\n",part_2refine);
	// printf("the 2project         %lf\n",part_2map);
	// printf("the slipt            %lf\n",part_slipt);
	// printf("the krefine          %10.2lf ms\n",part_krefine);
	// printf("the project          %10.2lf ms\n",part_map);
	// printf("the krefine_malloc   %10.2lf ms\n",part_mallocrefine);
    // printf("\n");
	
	print_time_coarsen();
    // coarsen_else = part_coarsen - (sinitcuda_match + scuda_match + sfindc2 + sinclusive_scan + sfindc4 + sexclusive_scan + sfind_cnvtxsedge_original + \
    //         sbb_segsort + sSort_cnedges_part1 + sinclusive_scan2 + sSort_cnedges_part2 + sSort_cnedges_part2_5 + sSort_cnedges_part3 + coarsen_malloc + \
    //         coarsen_memcpy + coarsen_free);

    // printf("initcuda_match             %10.3lf %7.3lf%\n", sinitcuda_match, sinitcuda_match / part_coarsen * 100);
    // printf("cuda_match                 %10.3lf %7.3lf%\n", scuda_match, scuda_match / part_coarsen * 100);
    // printf("findc2                     %10.3lf %7.3lf%\n", sfindc2, sfindc2 / part_coarsen * 100);
    // printf("prefix sum                 %10.3lf %7.3lf%\n", sinclusive_scan, sinclusive_scan / part_coarsen * 100);
    // printf("findc4                     %10.3lf %7.3lf%\n", sfindc4, sfindc4 / part_coarsen * 100);
    // printf("prefix sum                 %10.3lf %7.3lf%\n", sexclusive_scan, sexclusive_scan / part_coarsen * 100);
    // printf("find_cnvtxsedge_original   %10.3lf %7.3lf%\n", sfind_cnvtxsedge_original, sfind_cnvtxsedge_original / part_coarsen * 100);
    // printf("bb_segsort                 %10.3lf %7.3lf%\n", sbb_segsort, sbb_segsort / part_coarsen * 100);
    // printf("Sort_cnedges_part1         %10.3lf %7.3lf%\n", sSort_cnedges_part1, sSort_cnedges_part1 / part_coarsen * 100);
    // printf("prefix sum                 %10.3lf %7.3lf%\n", sinclusive_scan2, sinclusive_scan2 / part_coarsen * 100);
    // printf("Sort_cnedges_part2         %10.3lf %7.3lf%\n", sSort_cnedges_part2, sSort_cnedges_part2 / part_coarsen * 100);
    // printf("Sort_cnedges_part2_5       %10.3lf %7.3lf%\n", sSort_cnedges_part2_5, sSort_cnedges_part2_5 / part_coarsen * 100);
    // printf("Sort_cnedges_part3         %10.3lf %7.3lf%\n", sSort_cnedges_part3, sSort_cnedges_part3 / part_coarsen * 100);
    // printf("coarsen_malloc             %10.3lf %7.3lf%\n", coarsen_malloc, coarsen_malloc / part_coarsen * 100);
    // printf("coarsen_memcpy             %10.3lf %7.3lf%\n", coarsen_memcpy, coarsen_memcpy / part_coarsen * 100);
	// printf("coarsen_free               %10.3lf %7.3lf%\n", coarsen_free, coarsen_free / part_coarsen * 100);
    // printf("else                       %10.3lf %7.3lf%\n", coarsen_else, coarsen_else / part_coarsen * 100);

	// double twoway_else = gpu_2way - (initmoveto + updatemoveto + computepwgts + thrustreduce + computegain + thrustsort + computegainv + inclusive + re_balance);

    // printf("initmoveto=    %10.3lf %7.3lf%\n",initmoveto, initmoveto / gpu_2way * 100);
    // printf("updatemoveto=  %10.3lf %7.3lf%\n",updatemoveto, updatemoveto / gpu_2way * 100);
    // printf("computepwgts=  %10.3lf %7.3lf%\n",computepwgts, computepwgts / gpu_2way * 100);
    // printf("thrustreduce=  %10.3lf %7.3lf%\n",thrustreduce, thrustreduce / gpu_2way * 100);
    // printf("computegain=   %10.3lf %7.3lf%\n",computegain, computegain / gpu_2way * 100);
    // printf("thrustsort=    %10.3lf %7.3lf%\n",thrustsort, thrustsort / gpu_2way * 100);
    // printf("computegainv=  %10.3lf %7.3lf%\n",computegainv, computegainv / gpu_2way * 100);
    // printf("inclusive=     %10.3lf %7.3lf%\n",inclusive, inclusive / gpu_2way * 100);
    // printf("re_balance=    %10.3lf %7.3lf%\n",re_balance, re_balance / gpu_2way * 100);
	// printf("malloc_2way=   %10.3lf %7.3lf%\n",malloc_2way, malloc_2way / gpu_2way * 100);
    // printf("else=          %10.3lf %7.3lf%\n",twoway_else, twoway_else / gpu_2way * 100);
	// printf("\n");

	// double split_else = part_slipt - (malloc_split + memcpy_split + free_split);
	// printf("malloc_split=    %10.3lf %7.3lf%\n",malloc_split, malloc_split / part_slipt * 100);
	// printf("memcpy_split=    %10.3lf %7.3lf%\n",memcpy_split, memcpy_split / part_slipt * 100);
	// printf("free_split=      %10.3lf %7.3lf%\n",free_split, free_split / part_slipt * 100);
	// printf("else=            %10.3lf %7.3lf%\n",split_else, split_else / part_slipt * 100);
	// printf("\n");

	// save_init = save_init + memcpy_split;
	// printf("could_save=      %10.2lf %7.3lf%\n",save_init, save_init / part_init * 100);

	// print_time_uncoarsen();
	// printf("\n");
	// double Uncoarsen_else = part_uncoarsen - (krefine_atomicadd + uncoarsen_Sum_maxmin_pwgts + bndinfo_Find_real_bnd_info + bndinfo_init_bnd_info +\
	// 	bndinfo_find_kayparams + bndinfo_initcucsr + bndinfo_bb_segsort + bndinfo_init_cu_que + bndinfo_findcsr +\
	// 	uncoarsen_Exnode_part1 + uncoarsen_Exnode_part2);
	// printf("Uncoarsen Sumpwgts                     %10.3lf %.3f%\n",krefine_atomicadd, krefine_atomicadd / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen Sum_maxmin_pwgts   %10.3lf %.3f%\n",uncoarsen_Sum_maxmin_pwgts,uncoarsen_Sum_maxmin_pwgts / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen Find_real_bnd_info %10.3lf %.3f%\n",bndinfo_Find_real_bnd_info,bndinfo_Find_real_bnd_info / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen init_bnd_info      %10.3lf %.3f%\n",bndinfo_init_bnd_info,bndinfo_init_bnd_info / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen find_kayparams     %10.3lf %.3f%\n",bndinfo_find_kayparams,bndinfo_find_kayparams / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen initcucsr          %10.3lf %.3f%\n",bndinfo_initcucsr,bndinfo_initcucsr / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen bb_segsort         %10.3lf %.3f%\n",bndinfo_bb_segsort,bndinfo_bb_segsort / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen init_cu_que        %10.3lf %.3f%\n",bndinfo_init_cu_que,bndinfo_init_cu_que / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen findcsr            %10.3lf %.3f%\n",bndinfo_findcsr,bndinfo_findcsr / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen Exnode_part1       %10.3lf %.3f%\n",uncoarsen_Exnode_part1,uncoarsen_Exnode_part1 / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen Exnode_part2       %10.3lf %.3f%\n",uncoarsen_Exnode_part2,uncoarsen_Exnode_part2 / part_uncoarsen * 100);
	// printf("Uncoarsen uncoarsen else               %10.3lf %.3f%\n",Uncoarsen_else,Uncoarsen_else / part_uncoarsen * 100);

  // printf("match=                       %lf\n",part_match);
  // printf("contract=                    %lf\n",part_contract);
  // printf("cmatch=                      %lf\n",part_cmatch);
  // printf("ccontract=                   %lf\n",part_ccontract);
  // printf("bfs=                         %lf\n",part_bfs);
  // printf("2refine=                     %lf\n",part_2refine);
  // printf("2map=                        %lf\n",part_2map);
  // printf("slipt=                       %lf\n",part_slipt);
//   printf("krefine=                     %lf\n",part_krefine);
  // printf("map=                         %lf\n",part_map);
}

// repair cuda_nvtxs, cuda_nparts, refine_pass, cuda_tvwgt, Mallocinit_refineinfo, hunyuangraph_malloc_refineinfo, Sum_maxmin_pwgts, 
//     hunyuangraph_findgraphbndinfo: Find_real_bnd_info, find_kayparams, findcsr,
// repair Exnode_part1 Exnode_part2
// repair graph->cuda_real_edge, cuda_cnvtxs, cxadj, ccc, cuda_s, findc1, prefix_sum(cmap), findc22_53, finc4,
// repair cuda_js, find_cnvtxsedge_original? Sort_2Sort_2_5, Sort_cnedges2_part1, Sort_cnedges2_part3, setcvwgt, cuda_maxvwgt/maxvwgt, cuda_hem
// merge findc1 findc2, gpu_match 32->128, divide_group, merge findc4 set_cvwgt, cuda_scan_nedges_original, cuda_real_nvtxs, 
//error: Bump_2911(X), soc-Livejournal1�????

// 相较于cuMetis_10，本版本优化点如下：
// 		1、将cu_bn,cu_bt,cu_g,cu_csr,cu_que该五个变量的空间申请挪至k-way refinement阶段；
// 		2、粗化阶段中bb_segsort库函数中的cudaMalloc,cudaFree及cudaMemcpy；
// 		3、申请显存总空间进一步修改为“freeMem - usableMem - 2 * nedges * sizeof(int)”；

/*****************************		Attention about cudaMalloc!!!				*****************************
**********由于多级算法的对内存的要求与栈的性质完美的重合，因此本算法中涉及到的显存管理是以栈为基础的	   **********
**********由于算法中使用到了一些临时数组，仅使用单端栈的话会导致显存的浪费以及显存碎片化，			      **********
**********		因此本算法使用了双端栈的概念，左端存放需要保留的图的信息，右端存放临时数组				  **********
**********若利用本算法特定的显存管理方式，请注意“先申请后释放，后申请先释放”的特性						 **********
*/
// 左端栈先压入了cu_bn,cu_bt,cu_g,cu_csr,cu_que(cuMetis_10 yes/now no)
// 左端栈存放的图的信息从左至右依次为vwgt,xadj,adjncy,adjwgt,cmap,where,bnd,pwgts,tpwgts,maxwgt,minwgt,bndnum
// 粗化阶段右端栈存放的图的信息从右至左依次为match,txadj.tadjncy,tadjwgt,temp_scan