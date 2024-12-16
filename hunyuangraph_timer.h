#ifndef _H_TIME
#define _H_TIME

#include "hunyuangraph_struct.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_graph.h"

/*Time function params*/
// all time
double part_all = 0;
struct timeval begin_part_all;
struct timeval end_part_all;

// three phase
double part_coarsen = 0;
struct timeval begin_part_coarsen;
struct timeval end_part_coarsen;

double part_init = 0;
struct timeval begin_part_init;
struct timeval end_part_init;

double part_uncoarsen = 0;
struct timeval begin_part_uncoarsen;
struct timeval end_part_uncoarsen;

// steps
double part_match = 0;
struct timeval begin_part_match;
struct timeval end_part_match;

double part_contruction = 0;
struct timeval begin_part_contruction;
struct timeval end_part_contruction;

double part_cmatch = 0;
struct timeval begin_part_cmatch;
struct timeval end_part_cmatch;

double part_ccontract = 0;
struct timeval begin_part_ccontract;
struct timeval end_part_ccontract;

double part_bfs = 0;
struct timeval begin_part_bfs;
struct timeval end_part_bfs;

double part_2refine = 0;
struct timeval begin_part_2refine;
struct timeval end_part_2refine;

double part_2map = 0;
struct timeval begin_part_2map;
struct timeval end_part_2map;

double part_slipt = 0;
struct timeval begin_part_slipt;
struct timeval end_part_slipt;

double part_krefine = 0;
struct timeval begin_part_krefine;
struct timeval end_part_krefine;

double part_map = 0;
struct timeval begin_part_map;
struct timeval end_part_map;

double part_mallocrefine = 0;
struct timeval begin_part_mallocrefine;
struct timeval end_part_mallocrefine;

// test
double krefine_atomicadd = 0;
struct timeval begin_krefine_atomicadd;
struct timeval end_krefine_atomicadd;

// uncoarsen
double uncoarsen_Sum_maxmin_pwgts = 0;
double uncoarsen_Exnode_part1 = 0;
double uncoarsen_Exnode_part2 = 0;
struct timeval begin_general;
struct timeval end_general;

double bndinfo_Find_real_bnd_info = 0;
double bndinfo_init_bnd_info = 0;
double bndinfo_find_kayparams = 0;
double bndinfo_initcucsr = 0;
double bndinfo_bb_segsort = 0;
double bndinfo_init_cu_que = 0;
double bndinfo_findcsr = 0;
struct timeval begin_bndinfo;
struct timeval end_bndinfo;

// match
double init_gpu_match_time = 0;
double hem_gpu_match_time = 0;
double resolve_conflict_1_time = 0;
double resolve_conflict_2_time = 0;
double inclusive_scan_time1 = 0;
double resolve_conflict_4_time = 0;
struct timeval begin_gpu_match;
struct timeval end_gpu_match;

//match kernel
double random_match_time = 0;
double init_gpu_receive_send_time = 0;
double wgt_segmentsort_gpu_time = 0;
double segmentsort_memcpy_time = 0;
double set_receive_send_time = 0;
double set_match_topk_time = 0;
double reset_match_array_time = 0;
struct timeval begin_gpu_match_kernel;
struct timeval end_gpu_match_kernel;

// contract
double exclusive_scan_time = 0;
double set_tadjncy_tadjwgt_time = 0;
double ncy_segmentsort_gpu_time = 0;
double mark_edges_time = 0;
double inclusive_scan_time2 = 0;
double set_cxadj_time = 0;
double init_cadjwgt_time = 0;
double set_cadjncy_cadjwgt_time = 0;
struct timeval begin_gpu_contraction;
struct timeval end_gpu_contraction;

double coarsen_malloc = 0;
struct timeval begin_malloc;
struct timeval end_malloc;

double coarsen_memcpy = 0;
struct timeval begin_coarsen_memcpy;
struct timeval end_coarsen_memcpy;

double coarsen_free = 0;
struct timeval begin_free;
struct timeval end_free;

double coarsen_else = 0;

// init
double bisection_gpu_time = 0;
struct timeval begin_gpu_bisection;
struct timeval end_gpu_bisection;

double set_cpu_graph = 0;
struct timeval begin_set_cpu_graph;
struct timeval end_set_cpu_graph;

double set_graph_1 = 0;
double set_graph_2 = 0;
double set_graph_3 = 0;
double set_graph_4 = 0;
struct timeval begin_set_graph;
struct timeval end_set_graph;

double gpu_2way = 0;
struct timeval begin_gpu_2way;
struct timeval end_gpu_2way;

double malloc_2way = 0;
struct timeval begin_malloc_2way;
struct timeval end_malloc_2way;

double initmoveto = 0;
struct timeval begin_initmoveto;
struct timeval end_initmoveto;

double updatemoveto = 0;
struct timeval begin_updatemoveto;
struct timeval end_updatemoveto;

double computepwgts = 0;
struct timeval begin_computepwgts;
struct timeval end_computepwgts;

double thrustreduce = 0;
struct timeval begin_thrustreduce;
struct timeval end_thrustreduce;

double computegain = 0;
struct timeval begin_computegain;
struct timeval end_computegain;

double thrustsort = 0;
struct timeval begin_thrustsort;
struct timeval end_thrustsort;

double computegainv = 0;
struct timeval begin_computegainv;
struct timeval end_computegainv;

double inclusive = 0;
struct timeval begin_inclusive;
struct timeval end_inclusive;

double re_balance = 0;
struct timeval begin_rebalance;
struct timeval end_rebalance;

double malloc_split = 0;
struct timeval begin_malloc_split;
struct timeval end_malloc_split;

double memcpy_split = 0;
struct timeval begin_memcpy_split;
struct timeval end_memcpy_split;

double free_split = 0;
struct timeval begin_free_split;
struct timeval end_free_split;

double save_init = 0;
struct timeval begin_save_init;
struct timeval end_save_init;

void print_graph_infor(hunyuangraph_graph_t *graph, char *filename)
{
    printf("graph:%s %d %d\n", filename, graph->nvtxs, graph->nedges);
}

void init_timer()
{
    part_all = 0;
    part_coarsen = 0;
    part_init = 0;
    part_uncoarsen = 0;

    part_match = 0;
    part_contruction = 0;

    // match
    init_gpu_match_time = 0;
    hem_gpu_match_time = 0;
    resolve_conflict_1_time = 0;
    resolve_conflict_2_time = 0;
    inclusive_scan_time1 = 0;
    resolve_conflict_4_time = 0;

    //match kernel
    random_match_time = 0;
    init_gpu_receive_send_time = 0;
    wgt_segmentsort_gpu_time = 0;
    segmentsort_memcpy_time = 0;
    set_receive_send_time = 0;
    set_match_topk_time = 0;
    reset_match_array_time = 0;

    // contract
    exclusive_scan_time = 0;
    set_tadjncy_tadjwgt_time = 0;
    ncy_segmentsort_gpu_time = 0;
    mark_edges_time = 0;
    inclusive_scan_time2 = 0;
    set_cxadj_time = 0;
    init_cadjwgt_time = 0;
    set_cadjncy_cadjwgt_time = 0;
}

void print_time_all(hunyuangraph_graph_t *graph, int *part, int edgecut)
{
    printf("Hunyuangraph-Partition-end\n");
    printf("Hunyuangraph_Partition_time= %10.2lf ms\n", part_all);
    printf("------Coarsen_time=          %10.2lf ms\n", part_coarsen);
    printf("------Init_time=             %10.2lf ms\n", part_init);
    printf("------Uncoarsen_time=        %10.2lf ms\n", part_uncoarsen);
    printf("------else_time=             %10.2lf ms\n", part_all - (part_coarsen + part_init + part_uncoarsen));
    printf("edge-cut=                    %10d\n", edgecut);
}

void print_time_coarsen()
{
    printf("\n");

    coarsen_else = part_coarsen - (init_gpu_match_time + hem_gpu_match_time + resolve_conflict_1_time + resolve_conflict_2_time + inclusive_scan_time1 +
                                   resolve_conflict_4_time + exclusive_scan_time + set_tadjncy_tadjwgt_time + ncy_segmentsort_gpu_time + mark_edges_time + inclusive_scan_time2 +
                                   set_cxadj_time + init_cadjwgt_time + set_cadjncy_cadjwgt_time + coarsen_malloc + coarsen_memcpy + coarsen_free);
    printf("Coarsen_time=              %10.3lf ms\n", part_coarsen);
    printf("    part_match                 %10.3lf %7.3lf%\n", part_match, part_match / part_coarsen * 100);
    printf("        init_gpu_match_time        %10.3lf %7.3lf%\n", init_gpu_match_time, init_gpu_match_time / part_coarsen * 100);
    printf("        hem_gpu_match_time         %10.3lf %7.3lf%\n", hem_gpu_match_time, hem_gpu_match_time / part_coarsen * 100);
    printf("            random_match_time          %10.3lf %7.3lf%\n", random_match_time, random_match_time / hem_gpu_match_time * 100);
    printf("            init_gpu_receive_send_time %10.3lf %7.3lf%\n", init_gpu_receive_send_time, init_gpu_receive_send_time / hem_gpu_match_time * 100);
    printf("            wgt_segmentsort_gpu_time   %10.3lf %7.3lf%\n", wgt_segmentsort_gpu_time, wgt_segmentsort_gpu_time / hem_gpu_match_time * 100);
    printf("            segmentsort_memcpy_time    %10.3lf %7.3lf%\n", segmentsort_memcpy_time, segmentsort_memcpy_time / hem_gpu_match_time * 100);
    printf("            set_receive_send_time      %10.3lf %7.3lf%\n", set_receive_send_time, set_receive_send_time / hem_gpu_match_time * 100);
    printf("            set_match_topk_time        %10.3lf %7.3lf%\n", set_match_topk_time, set_match_topk_time / hem_gpu_match_time * 100);
    printf("            reset_match_array_time     %10.3lf %7.3lf%\n", reset_match_array_time, reset_match_array_time / hem_gpu_match_time * 100);
    printf("        resolve_conflict_1_time    %10.3lf %7.3lf%\n", resolve_conflict_1_time, resolve_conflict_1_time / part_coarsen * 100);
    printf("        resolve_conflict_2_time    %10.3lf %7.3lf%\n", resolve_conflict_2_time, resolve_conflict_2_time / part_coarsen * 100);
    printf("        inclusive_scan_time        %10.3lf %7.3lf%\n", inclusive_scan_time1, inclusive_scan_time1 / part_coarsen * 100);
    printf("        resolve_conflict_4_time    %10.3lf %7.3lf%\n", resolve_conflict_4_time, resolve_conflict_4_time / part_coarsen * 100);
    printf("    part_contruction           %10.3lf %7.3lf%\n", part_contruction, part_contruction / part_coarsen * 100);
    printf("        exclusive_scan_time        %10.3lf %7.3lf%\n", exclusive_scan_time, exclusive_scan_time / part_coarsen * 100);
    printf("        set_tadjncy_tadjwgt_time   %10.3lf %7.3lf%\n", set_tadjncy_tadjwgt_time, set_tadjncy_tadjwgt_time / part_coarsen * 100);
    printf("        ncy_segmentsort_gpu_time   %10.3lf %7.3lf%\n", ncy_segmentsort_gpu_time, ncy_segmentsort_gpu_time / part_coarsen * 100);
    printf("        mark_edges_time            %10.3lf %7.3lf%\n", mark_edges_time, mark_edges_time / part_coarsen * 100);
    printf("        inclusive_scan_time2       %10.3lf %7.3lf%\n", inclusive_scan_time2, inclusive_scan_time2 / part_coarsen * 100);
    printf("        set_cxadj_time             %10.3lf %7.3lf%\n", set_cxadj_time, set_cxadj_time / part_coarsen * 100);
    printf("        init_cadjwgt_time          %10.3lf %7.3lf%\n", init_cadjwgt_time, init_cadjwgt_time / part_coarsen * 100);
    printf("        set_cadjncy_cadjwgt_time   %10.3lf %7.3lf%\n", set_cadjncy_cadjwgt_time, set_cadjncy_cadjwgt_time / part_coarsen * 100);
    printf("    coarsen_malloc             %10.3lf %7.3lf%\n", coarsen_malloc, coarsen_malloc / part_coarsen * 100);
    printf("    coarsen_memcpy             %10.3lf %7.3lf%\n", coarsen_memcpy, coarsen_memcpy / part_coarsen * 100);
    printf("    coarsen_free               %10.3lf %7.3lf%\n", coarsen_free, coarsen_free / part_coarsen * 100);
    printf("    else                       %10.3lf %7.3lf%\n", coarsen_else, coarsen_else / part_coarsen * 100);
}

void print_time_init()
{
    printf("\n");

    printf("        gpu_Bisection_time         %10.3lf %7.3lf%\n", bisection_gpu_time, bisection_gpu_time / bisection_gpu_time * 100);
}

void print_time_uncoarsen()
{
    printf("\n");
    double Uncoarsen_else = part_uncoarsen - (krefine_atomicadd + uncoarsen_Sum_maxmin_pwgts + bndinfo_Find_real_bnd_info + bndinfo_init_bnd_info +
                                              bndinfo_find_kayparams + bndinfo_initcucsr + bndinfo_bb_segsort + bndinfo_init_cu_que + bndinfo_findcsr +
                                              uncoarsen_Exnode_part1 + uncoarsen_Exnode_part2);
    printf("Uncoarsen Sumpwgts                     %10.3lf %.3f%\n", krefine_atomicadd, krefine_atomicadd / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen Sum_maxmin_pwgts   %10.3lf %.3f%\n", uncoarsen_Sum_maxmin_pwgts, uncoarsen_Sum_maxmin_pwgts / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen Find_real_bnd_info %10.3lf %.3f%\n", bndinfo_Find_real_bnd_info, bndinfo_Find_real_bnd_info / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen init_bnd_info      %10.3lf %.3f%\n", bndinfo_init_bnd_info, bndinfo_init_bnd_info / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen find_kayparams     %10.3lf %.3f%\n", bndinfo_find_kayparams, bndinfo_find_kayparams / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen initcucsr          %10.3lf %.3f%\n", bndinfo_initcucsr, bndinfo_initcucsr / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen bb_segsort         %10.3lf %.3f%\n", bndinfo_bb_segsort, bndinfo_bb_segsort / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen init_cu_que        %10.3lf %.3f%\n", bndinfo_init_cu_que, bndinfo_init_cu_que / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen findcsr            %10.3lf %.3f%\n", bndinfo_findcsr, bndinfo_findcsr / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen Exnode_part1       %10.3lf %.3f%\n", uncoarsen_Exnode_part1, uncoarsen_Exnode_part1 / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen Exnode_part2       %10.3lf %.3f%\n", uncoarsen_Exnode_part2, uncoarsen_Exnode_part2 / part_uncoarsen * 100);
    printf("Uncoarsen uncoarsen else               %10.3lf %.3f%\n", Uncoarsen_else, Uncoarsen_else / part_uncoarsen * 100);
}

#endif