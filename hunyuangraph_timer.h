#ifndef _H_TIME
#define _H_TIME

#include "hunyuangraph_struct.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_graph.h"

#include <cuda_runtime.h>

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
double uncoarsen_Exnode_part1 = 0;
double uncoarsen_Exnode_part2 = 0;
struct timeval begin_general;
struct timeval   end_general;

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
double check_length_time = 0;
double set_bin_time = 0;
double hem_gpu_match_time = 0;
double resolve_conflict_1_time = 0;
double resolve_conflict_2_time = 0;
double inclusive_scan_time1 = 0;
double resolve_conflict_4_time = 0;
struct timeval begin_gpu_match;
struct timeval end_gpu_match;

//match kernel
double match_time = 0;
double random_match_time = 0;
double init_gpu_receive_send_time = 0;
double wgt_segmentsort_gpu_time = 0;
double segmentsort_memcpy_time = 0;
double set_receive_send_time = 0;
double set_match_topk_time = 0;
double reset_match_array_time = 0;
double leaf_matches_step1_time = 0;
double leaf_matches_step2_time = 0;
double isolate_matches_time = 0;
double twin_matches_time = 0;
double relative_matches_step1_time = 0;
double relative_matches_step2_time = 0;
double match_malloc_time = 0;
double match_memcpy_time = 0;
double match_free_time = 0;
struct timeval begin_gpu_match_kernel;
struct timeval end_gpu_match_kernel;

//topk / four match time
double top1_time = 0;
double top2_time = 0;
double top3_time = 0;
double top4_time = 0;
double leaf_time = 0;
double isolate_time = 0;
double twin_time = 0;
double relative_time = 0;
struct timeval begin_gpu_topkfour_match;
struct timeval end_gpu_topkfour_match;

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
double set_initgraph_time = 0;
double initcurand_gpu_time = 0;
double bisection_gpu_time = 0;
double splitgraph_gpu_time = 0;
double select_where_gpu_time = 0;
double update_where_gpu_time = 0;
double update_answer_gpu_time = 0;
double update_tpwgts_time = 0;
double computecut_time = 0;
struct timeval begin_gpu_bisection;
struct timeval end_gpu_bisection;

double init_else = 0;

//  uncoarsen
double uncoarsen_initpwgts = 0;
double uncoarsen_calculateSum = 0;
double uncoarsen_projectback = 0;
double uncoarsen_Sum_maxmin_pwgts = 0;
double uncoarsen_select_init_select = 0;
double uncoarsen_select_bnd_vertices_warp = 0;
double uncoarsen_moving_interaction = 0;
double uncoarsen_update_select = 0;
double uncoarsen_execute_move = 0;
double uncoarsen_compute_edgecut = 0;
double uncoarsen_gpu_malloc = 0;
double uncoarsen_gpu_free = 0;
struct timeval begin_gpu_kway;
struct timeval end_gpu_kway;
double uncoarsen_lp = 0;
double uncoarsen_rw = 0;
double uncoarsen_rs = 0;
double uncoarsen_pm = 0;

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

    //  coarsen
    init_gpu_match_time = 0;
    check_length_time = 0;
    set_bin_time = 0;
    hem_gpu_match_time = 0;
    resolve_conflict_1_time = 0;
    resolve_conflict_2_time = 0;
    inclusive_scan_time1 = 0;
    resolve_conflict_4_time = 0;
    match_malloc_time = 0;
    match_memcpy_time = 0;
    match_free_time = 0;
    exclusive_scan_time = 0;
    set_tadjncy_tadjwgt_time = 0;
    ncy_segmentsort_gpu_time = 0;
    mark_edges_time = 0;
    inclusive_scan_time2 = 0;
    set_cxadj_time = 0;
    init_cadjwgt_time = 0;
    set_cadjncy_cadjwgt_time = 0;
    coarsen_malloc = 0;
    coarsen_memcpy = 0;
    coarsen_free = 0;

    // topk
    top1_time = 0;
    top2_time = 0;
    top3_time = 0;
    top4_time = 0;
    leaf_time = 0;
    isolate_time = 0;
    twin_time = 0;
    relative_time = 0;

    //  init
    set_initgraph_time = 0;
    initcurand_gpu_time = 0;
    bisection_gpu_time = 0;
    splitgraph_gpu_time = 0;
    select_where_gpu_time = 0;
    update_where_gpu_time = 0;
    update_answer_gpu_time = 0;
    update_tpwgts_time = 0;

    // uncoarsen
    uncoarsen_initpwgts = 0;
    uncoarsen_calculateSum = 0;
    uncoarsen_Sum_maxmin_pwgts = 0;
    uncoarsen_select_init_select = 0;
    uncoarsen_moving_interaction = 0;
    uncoarsen_update_select = 0;
    uncoarsen_execute_move = 0;
    uncoarsen_select_bnd_vertices_warp = 0;
    uncoarsen_projectback = 0;
    uncoarsen_gpu_malloc = 0;
    uncoarsen_gpu_free = 0;
    uncoarsen_compute_edgecut = 0;

    uncoarsen_lp = 0;
    uncoarsen_rw = 0;
    uncoarsen_rs = 0;
    uncoarsen_pm = 0;
}

void print_time_all(hunyuangraph_graph_t *graph, int *part, int edgecut, float imbalance)
{
    printf("---------------------------------------------------------\n");
    printf("Hunyuangraph-Partition-end\n");
    printf("Hunyuangraph_Partition_time= %10.2lf ms\n", part_all);
    printf("------Coarsen_time=          %10.2lf ms\n", part_coarsen);
    printf("------Init_time=             %10.2lf ms\n", part_init);
    printf("------Uncoarsen_time=        %10.2lf ms\n", part_uncoarsen);
    printf("------else_time=             %10.2lf ms\n", part_all - (part_coarsen + part_init + part_uncoarsen));
    printf("edge-cut=                    %10d\n", edgecut);
    printf("imbalance=                   %10.3f\n", imbalance);
    printf("---------------------------------------------------------\n");
}

void print_time_coarsen()
{
    printf("\n");

    coarsen_else = part_coarsen - (init_gpu_match_time + check_length_time + set_bin_time + hem_gpu_match_time + resolve_conflict_1_time + resolve_conflict_2_time + inclusive_scan_time1 +
                                        resolve_conflict_4_time - match_malloc_time - match_memcpy_time - match_free_time + \
                                   exclusive_scan_time + set_tadjncy_tadjwgt_time + ncy_segmentsort_gpu_time + mark_edges_time + inclusive_scan_time2 +
                                    set_cxadj_time + init_cadjwgt_time + set_cadjncy_cadjwgt_time + coarsen_malloc + coarsen_memcpy + coarsen_free);
    printf("---------------------------------------------------------\n");
    printf("Coarsen_time=              %10.3lf ms\n", part_coarsen);
    printf("    part_match                 %10.3lf %7.3lf%\n", part_match, part_match / part_coarsen * 100);
    printf("        init_gpu_match_time        %10.3lf %7.3lf%\n", init_gpu_match_time, init_gpu_match_time / part_coarsen * 100);
    printf("        check_length_time          %10.3lf %7.3lf%\n", check_length_time, check_length_time / part_coarsen * 100);
    printf("        set_bin_time               %10.3lf %7.3lf%\n", set_bin_time, set_bin_time / part_coarsen * 100);
    printf("        hem_gpu_match_time         %10.3lf %7.3lf%\n", hem_gpu_match_time, hem_gpu_match_time / part_coarsen * 100);
    printf("            random_match_time          %10.3lf %7.3lf%\n", random_match_time, random_match_time / hem_gpu_match_time * 100);
    printf("            init_gpu_receive_send_time %10.3lf %7.3lf%\n", init_gpu_receive_send_time, init_gpu_receive_send_time / hem_gpu_match_time * 100);
    printf("            wgt_segmentsort_gpu_time   %10.3lf %7.3lf%\n", wgt_segmentsort_gpu_time, wgt_segmentsort_gpu_time / hem_gpu_match_time * 100);
    printf("            segmentsort_memcpy_time    %10.3lf %7.3lf%\n", segmentsort_memcpy_time, segmentsort_memcpy_time / hem_gpu_match_time * 100);
    printf("            set_receive_send_time      %10.3lf %7.3lf%\n", set_receive_send_time, set_receive_send_time / hem_gpu_match_time * 100);
    printf("            set_match_topk_time        %10.3lf %7.3lf%\n", set_match_topk_time, set_match_topk_time / hem_gpu_match_time * 100);
    printf("            reset_match_array_time     %10.3lf %7.3lf%\n", reset_match_array_time, reset_match_array_time / hem_gpu_match_time * 100);
    printf("            leaf_matches               %10.3lf %7.3lf%\n", leaf_matches_step1_time + leaf_matches_step2_time, (leaf_matches_step1_time + leaf_matches_step2_time) / hem_gpu_match_time * 100);
    printf("                step1                      %10.3lf %7.3lf%\n", leaf_matches_step1_time, leaf_matches_step1_time / hem_gpu_match_time * 100);
    printf("                step2                      %10.3lf %7.3lf%\n", leaf_matches_step2_time, leaf_matches_step2_time / hem_gpu_match_time * 100);
    printf("            isolate_matches            %10.3lf %7.3lf%\n", isolate_matches_time, isolate_matches_time / hem_gpu_match_time * 100);
    printf("            twin_matches               %10.3lf %7.3lf%\n", twin_matches_time, twin_matches_time / hem_gpu_match_time * 100);
    printf("            relative_matches           %10.3lf %7.3lf%\n", relative_matches_step1_time + relative_matches_step2_time, (relative_matches_step1_time + relative_matches_step2_time) / hem_gpu_match_time * 100);
    printf("                step1                      %10.3lf %7.3lf%\n", relative_matches_step1_time, relative_matches_step1_time / hem_gpu_match_time * 100);
    printf("                step2                      %10.3lf %7.3lf%\n", relative_matches_step2_time, relative_matches_step2_time / hem_gpu_match_time * 100);
    printf("            match_malloc_time          %10.3lf %7.3lf%\n", match_malloc_time, match_malloc_time / hem_gpu_match_time * 100);
    printf("            match_memcpy_time          %10.3lf %7.3lf%\n", match_memcpy_time, match_memcpy_time / hem_gpu_match_time * 100);
    printf("            match_free_time            %10.3lf %7.3lf%\n", match_free_time, match_free_time / hem_gpu_match_time * 100);
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
    printf("---------------------------------------------------------\n");
}

void print_time_topkfour_match()
{
    double all = top1_time + top2_time + top3_time + top4_time + leaf_time + isolate_time + twin_time + relative_time;
    
    printf("---------------------------------------------------------\n");
    printf("all                        %10.3lf ms\n", all);
    printf("top1_time                  %10.3lf %7.3lf%\n", top1_time, top1_time / all * 100);
    printf("top2_time                  %10.3lf %7.3lf%\n", top2_time, top2_time / all * 100);
    printf("top3_time                  %10.3lf %7.3lf%\n", top3_time, top3_time / all * 100);
    printf("top4_time                  %10.3lf %7.3lf%\n", top4_time, top4_time / all * 100);
    printf("leaf_time                  %10.3lf %7.3lf%\n", leaf_time, leaf_time / all * 100);
    printf("isolate_time               %10.3lf %7.3lf%\n", isolate_time, isolate_time / all * 100);
    printf("twin_time                  %10.3lf %7.3lf%\n", twin_time, twin_time / all * 100);
    printf("relative_time              %10.3lf %7.3lf%\n", relative_time, relative_time / all * 100);
    printf("---------------------------------------------------------\n");
}

void print_time_init()
{
    printf("\n");

    init_else = part_init - (set_initgraph_time + initcurand_gpu_time + bisection_gpu_time + splitgraph_gpu_time + select_where_gpu_time + update_where_gpu_time + update_answer_gpu_time + 
                             update_tpwgts_time);
    
    printf("---------------------------------------------------------\n");
    printf("Init_time=                 %10.3lf ms\n", part_init);
    printf("    set_initgraph_time         %10.3lf %7.3lf%\n", set_initgraph_time, set_initgraph_time / part_init * 100);
    printf("    update_tpwgts_time         %10.3lf %7.3lf%\n", update_tpwgts_time, update_tpwgts_time / part_init * 100);
    printf("    initcurand_gpu_time        %10.3lf %7.3lf%\n", initcurand_gpu_time, initcurand_gpu_time / part_init * 100);
    printf("    gpu_Bisection_time         %10.3lf %7.3lf%\n", bisection_gpu_time, bisection_gpu_time / part_init * 100);
    printf("    splitgraph_gpu_time        %10.3lf %7.3lf%\n", splitgraph_gpu_time, splitgraph_gpu_time / part_init * 100);
    printf("    select_where_gpu_time      %10.3lf %7.3lf%\n", select_where_gpu_time, select_where_gpu_time / part_init * 100);
    printf("    update_where_gpu_time      %10.3lf %7.3lf%\n", update_where_gpu_time, update_where_gpu_time / part_init * 100);
    printf("    update_answer_gpu_time     %10.3lf %7.3lf%\n", update_answer_gpu_time, update_answer_gpu_time / part_init * 100);
    printf("    else                       %10.3lf %7.3lf%\n", init_else, init_else / part_init * 100);

    // printf("    computecut_time            %10.3lf %7.3lf%\n", computecut_time, computecut_time / part_init * 100);
    printf("---------------------------------------------------------\n");
}

void print_time_uncoarsen()
{
    printf("\n");
    double Uncoarsen_else = part_uncoarsen - (uncoarsen_lp + uncoarsen_rw + uncoarsen_rs + uncoarsen_pm + \
                                              uncoarsen_projectback + uncoarsen_gpu_malloc + uncoarsen_gpu_free + uncoarsen_compute_edgecut);

    printf("---------------------------------------------------------\n");
    printf("Uncoarsen_time=            %10.3lf ms\n", part_uncoarsen);
    printf("Uncoarsen lp                   %10.3lf %7.3lf%\n", uncoarsen_lp, uncoarsen_lp / part_uncoarsen * 100);
    printf("Uncoarsen rw                   %10.3lf %7.3lf%\n", uncoarsen_rw, uncoarsen_rw / part_uncoarsen * 100);
    printf("Uncoarsen rs                   %10.3lf %7.3lf%\n", uncoarsen_rs, uncoarsen_rs / part_uncoarsen * 100);
    printf("Uncoarsen pm                   %10.3lf %7.3lf%\n", uncoarsen_pm, uncoarsen_pm / part_uncoarsen * 100);
    printf("Uncoarsen projectback          %10.3lf %7.3lf%\n", uncoarsen_projectback, uncoarsen_projectback / part_uncoarsen * 100);
    printf("Uncoarsen malloc               %10.3lf %7.3lf%\n", uncoarsen_gpu_malloc, uncoarsen_gpu_malloc / part_uncoarsen * 100);
    printf("Uncoarsen free                 %10.3lf %7.3lf%\n", uncoarsen_gpu_free, uncoarsen_gpu_free / part_uncoarsen * 100);
    printf("Uncoarsen compute_edgecut      %10.3lf %7.3lf%\n", uncoarsen_compute_edgecut, uncoarsen_compute_edgecut / part_uncoarsen * 100);
    printf("Uncoarsen else                 %10.3lf %7.3lf%\n", Uncoarsen_else, Uncoarsen_else / part_uncoarsen * 100);
    printf("---------------------------------------------------------\n");

    // printf("Uncoarsen initpwgts            %10.3lf %7.3lf%\n", uncoarsen_initpwgts, uncoarsen_initpwgts / part_uncoarsen * 100);
    // printf("Uncoarsen calculateSum         %10.3lf %7.3lf%\n", uncoarsen_calculateSum, uncoarsen_calculateSum / part_uncoarsen * 100);
    // printf("Uncoarsen Sum_maxmin_pwgts     %10.3lf %7.3lf%\n", uncoarsen_Sum_maxmin_pwgts, uncoarsen_Sum_maxmin_pwgts / part_uncoarsen * 100);
    // printf("Uncoarsen select_init          %10.3lf %7.3lf%\n", uncoarsen_select_init_select, uncoarsen_select_init_select / part_uncoarsen * 100);
    // printf("Uncoarsen select_bnd           %10.3lf %7.3lf%\n", uncoarsen_select_bnd_vertices_warp, uncoarsen_select_bnd_vertices_warp / part_uncoarsen * 100);
    // printf("Uncoarsen moving_interaction   %10.3lf %7.3lf%\n", uncoarsen_moving_interaction, uncoarsen_moving_interaction / part_uncoarsen * 100);
    // printf("Uncoarsen update_select        %10.3lf %7.3lf%\n", uncoarsen_update_select, uncoarsen_update_select / part_uncoarsen * 100);
    // printf("Uncoarsen execute_move         %10.3lf %7.3lf%\n", uncoarsen_execute_move, uncoarsen_execute_move / part_uncoarsen * 100);
}

#endif