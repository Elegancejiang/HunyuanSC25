#ifndef _H_STRUCT
#define _H_STRUCT

#include <sys/types.h>

typedef signed char hunyuangraph_int8_t;

/*Graph data structure*/
typedef struct hunyuangraph_graph_t {
	/*graph cpu params*/
	int nvtxs;                            //Graph vertex
	int nedges;	                          //Graph edge
	int *xadj;                            //Graph vertex csr array (xadj[nvtxs+1])
	int *adjncy;                          //Graph adjacency list (adjncy[nedges])
	int *adjwgt;   		                    //Graph edge weight array (adjwgt[nedges])
	int *vwgt;			                      //Graph vertex weight array(vwgr[nvtxs])
	int *tvwgt;                           //The sum of graph vertex weight 
	float *tvwgt_reverse;                 //The reciprocal of tvwgt
	int *label;                           //Graph vertex label(label[nvtxs])
	int *cmap;                            //The Label of graph vertex in cgraph(cmap[nvtxs]) 
	int mincut;                           //The min edfe-cut of graph partition
	int *where;                           //The label of graph vertex in which part(where[nvtxs]) 
	int *pwgts;                           //The partition vertex weight(pwgts[nparts])
	int nbnd;                             //Boundary vertex number
	int *bndlist;                         //Boundary vertex list
	int *bndptr;                          //Boundary vertex pointer
	int *id;                              //The sum of edge weight in same part
	int *ed;                              //The sum of edge weight in different part
	int ncon;
	struct hunyuangraph_graph_t *coarser; //The coarser graph
	struct hunyuangraph_graph_t *finer;   //The finer graph
	/*graph gpu params*/
	int *cuda_xadj;
	int *cuda_adjncy;
	int *cuda_adjwgt;
	int *cuda_vwgt;               
	int *cuda_match;                      //CUDA graph vertex match array(match[nvtxs])
	int *cuda_cmap;
	// int *cuda_maxvwgt;                    //CUDA graph constraint vertex weight 
	int *txadj;                  //CUDA graph vertex pairs csr edge array(txadj[cnvtxs+1])
	//   int *cuda_real_nvtxs;                 //CUDA graph params (i<match[cmap[i]])
	//   int *cuda_s;                          //CUDA support array (cuda_s[nvtxs])
	int *tadjwgt;       //CUDA support scan array (tadjwgt[nedges])
	int *bin_offset;
	int *bin_size;
	int *bin_rowidx;
	//   int *cuda_scan_nedges_original;       //CUDA support scan array (cuda_scan_nedges_original[nedges])
	int *tadjncy;      //CUDA support scan array (tadjncy[nedges])
	int *cuda_maxwgt;                     //CUDA part weight array (cuda_maxwgt[npart])
	int *cuda_minwgt;                     //CUDA part weight array (cuda_minwgt[npart])
	int *cuda_where;
	int *cuda_label;
	int *cuda_pwgts;
	int *cuda_ed;
	int *cuda_id;
	int *cuda_bndlist;
	int *cuda_bnd;
	int *cuda_bndnum;
	int *cpu_bndnum;
	int *cuda_info;                       //CUDA support array(cuda_info[bnd_num*nparts])
	int *cuda_real_bnd_num;
	int *cuda_real_bnd;
	//   int *cuda_tvwgt;  // graph->tvwgt
	float *cuda_tpwgts;

	/*Refinement available generate array*/
	int *cuda_balance;
	int *cuda_bn;                             
	int *cuda_to;
	int *cuda_gain;		//	???
	char *cuda_select;
	int *cuda_csr;
	int *cuda_que;
} hunyuangraph_graph_t;

/*Memory allocation information*/
typedef struct hunyuangraph_mop_t {
  int type;
  ssize_t nbytes;
  void *ptr;
} hunyuangraph_mop_t;

/*Algorithm information*/
typedef struct hunyuangraph_mcore_t {
  void *core;	
  size_t coresize;     
  size_t corecpos;            
  size_t nmops;         
  size_t cmop;         
  hunyuangraph_mop_t *mops;      
  size_t num_callocs;   
  size_t num_hallocs;   
  size_t size_callocs;  
  size_t size_hallocs;  
  size_t cur_callocs;   
  size_t cur_hallocs;  
  size_t max_callocs;   
  size_t max_hallocs;   

} hunyuangraph_mcore_t;

/*Control information*/
typedef struct hunyuangraph_admin_t {
  int Coarsen_threshold;		
  int nIparts;      
  int no2hop;                                                                                                                                 
  int iteration_num;                               
  int maxvwgt;		                
  int nparts;
  int ncuts;
  float *ubfactors;            
  float *tpwgts;               
  float *part_balance;               
  float cfactor;               
  hunyuangraph_mcore_t *mcore;    
  size_t nbrpoolsize;      
  size_t nbrpoolcpos;                  

} hunyuangraph_admin_t;

/*Heap information*/
typedef struct hunyuangraph_rkv_t{
  float key;
  int val;
} hunyuangraph_rkv_t;

/*Queue information*/
typedef struct {
	ssize_t nnodes;
	ssize_t maxnodes;
	hunyuangraph_rkv_t   *heap;
	ssize_t *locator;
} hunyuangraph_queue_t;

typedef struct 
{
	int nownodes;
	int maxnodes;
	int *key;
	int *val;
	int *locator;
} priority_queue_t;

typedef struct {
  int key;
  int ptr;
} kp_t;

typedef struct ikv_t{
  int key;
  int val;
} ikv_t;

#endif