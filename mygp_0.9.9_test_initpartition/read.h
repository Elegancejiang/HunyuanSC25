#ifndef READ_H
#define READ_H

#include <stdio.h>  // NULL
#include <sys/stat.h>   // stat
#include <string.h> // strcat
#include "struct.h"
#include "define.h"
#include "common.h"
#include "memory.h"
#include "graph.h"

/*************************************************************************
* This function checks if a file exists
**************************************************************************/
Hunyuan_int_t Is_file_exists(char *fname)
{
    struct stat status;

    if (stat(fname, &status) == -1)
        return 0;

    return S_ISREG(status.st_mode);
}

/*************************************************************************
* This function opens a file
**************************************************************************/
FILE *check_fopen(char *fname, char *mode, const char *message)
{
    FILE *fp = fopen(fname, mode);
    if (fp != NULL)
        return fp;

    char *error_message = (char *)check_malloc(sizeof(char) * 1024, "check_fopen: error_message");
	sprintf(error_message, "Failed on check_fopen()\n\tfile: %s, mode: %s, [ %s ].", fname, mode, message);
    error_exit(error_message);

    return NULL;
}

/*************************************************************************/
/*! This function is the GKlib implementation of glibc's getline()
    function.
    \returns -1 if the EOF has been reached, otherwise it returns the 
             number of bytes read.
*/
/*************************************************************************/
ssize_t check_getline(char **lineptr, size_t *n, FILE *stream)
{
    size_t i;
    Hunyuan_int_t ch;

    /* Check whether the file stream reaches the end of the file, and if it does, return -1 */
    if (feof(stream))
        return -1;  

    /* Initial memory allocation if *lineptr is NULL */
    if (*lineptr == NULL || *n == 0) {
        *n = 1024;
        *lineptr = (char *)check_malloc(sizeof(char) * (*n), "check_getline: lineptr");
    }

    /* get Hunyuan_int_to the main loop */
    i = 0;
    /* The getc function is used to read characters from the file stream until the end of the file is reached */
    while ((ch = getc(stream)) != EOF) {
        (*lineptr)[i++] = (char)ch;

        /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
        if (i+1 == *n) { 
            *n = 2*(*n);
            *lineptr = (char *)check_realloc(*lineptr, (*n)*sizeof(char), (*n)*sizeof(char)/2, "check_getline: lineptr");
        }
        
        if (ch == '\n')
            break;
    }
    (*lineptr)[i] = '\0';

    return (i == 0 ? -1 : i);
}

/*************************************************************************
* This function closes a file
**************************************************************************/
void check_fclose(FILE *fp)
{
	fclose(fp);
}

/*************************************************************************/
/*! This function reads in a sparse graph */
/*************************************************************************/
/*
params->filename = graphfile

*/
graph_t *ReadGraph(char *filename, Hunyuan_int_t **txadj, Hunyuan_int_t **tvwgt, Hunyuan_int_t **tadjncy, Hunyuan_int_t **tadjwgt)
{
    Hunyuan_int_t i, j, k, l, fmt, nfields, readew, readvw, readvs, edge, ewgt;
    Hunyuan_int_t *xadj, *adjncy, *vwgt, *adjwgt;
	Hunyuan_int_t *vsize;
    char *line = NULL, fmtstr[256], *curstr, *newstr;
    size_t lnlen = 0;
    FILE *fpin;
    graph_t *graph;

    if (!Is_file_exists(filename)) 
    {
        // char *error_message = (char *)malloc(sizeof(char) * 128);   //!!!
        char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
		sprintf(error_message, "File %s does not exist!", filename);
        error_exit(error_message);
        // errexit("File %s does not exist!\n", params->filename);
    }
	
    graph = CreateGraph();

    fpin = check_fopen(filename, "r", "ReadGRaph: Graph");
	
    /* Skip comment lines until you get to the first valid line */
    do {
        if (check_getline(&line, &lnlen, fpin) == -1) 
        {
            char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
			sprintf(error_message, "Premature end of input file: file: %s.", filename);
            error_exit(error_message);
        }
    } while (line[0] == '%');

    fmt = 0;
    nfields = sscanf(line, "%"PRIDX" %"PRIDX" %"PRIDX"", &(graph->nvtxs), &(graph->nedges), &fmt);
	
	if (nfields < 2) 
	{
		char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
		sprintf(error_message, "The input file does not specify the number of vertices and edges.");
		error_exit(error_message);
	}
	
	if (graph->nvtxs <= 0 || graph->nedges <= 0) 
	{
		char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
		sprintf(error_message, "The supplied nvtxs: %"PRIDX" and nedges: %"PRIDX" must be positive.", graph->nvtxs, graph->nedges);
		error_exit(error_message);
	}
	
	if (fmt > 111) 
	{
		char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
		sprintf(error_message, "Cannot read this type of file format [fmt=%"PRIDX"]!", fmt);
		error_exit(error_message);
	}
	
	sprintf(fmtstr, "%03"PRIDX"", fmt%1000);
	readvs = (fmtstr[0] == '1');
	readvw = (fmtstr[1] == '1');
	readew = (fmtstr[2] == '1');
	
	// if (ncon > 0 && !readvw) 
	// {
	// 	char *error_message = (char *)check_malloc(sizeof(char) * 1024, "ReadGraph: error_message");
	// 	sprintf(error_message, 
	// 	"------------------------------------------------------------------------------\n"
	// 	"***  I detected an error in your input file  ***\n\n"
	// 	"You specified ncon=%d, but the fmt parameter does not specify vertex weights\n" 
	// 	"Make sure that the fmt parameter is set to either 10 or 11.\n"
	// 	"------------------------------------------------------------------------------\n", ncon);
	// 	error_exit(error_message);
	// }

	graph->nedges *=2;
	// ncon = graph->ncon = (ncon == 0 ? 1 : ncon);

	// xadj   = graph->xadj   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (graph->nvtxs + 1), "ReadGraph: xadj");
	// adjncy = graph->adjncy = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "ReadGraph: adjncy");
	// vwgt   = graph->vwgt   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "ReadGraph: vwgt");
	// adjwgt = graph->adjwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges,"ReadGraph: adjwgt");
	xadj   = *txadj   = (Hunyuan_int_t *)malloc(sizeof(Hunyuan_int_t) * (graph->nvtxs + 1));
	adjncy = *tadjncy = (Hunyuan_int_t *)malloc(sizeof(Hunyuan_int_t) * graph->nedges);
	vwgt   = *tvwgt   = (Hunyuan_int_t *)malloc(sizeof(Hunyuan_int_t) * graph->nvtxs);
	adjwgt = *tadjwgt = (Hunyuan_int_t *)malloc(sizeof(Hunyuan_int_t) * graph->nedges);

	// vsize  = graph->vsize  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs,"ReadGraph: vsize");

	set_value_int(graph->nvtxs + 1, 0, xadj);
	set_value_int(graph->nvtxs, 1, vwgt);
	set_value_int(graph->nedges, 1, adjwgt);
	// set_value_int(graph->nvtxs, 1, vsize);

	/*----------------------------------------------------------------------
	* Read the sparse graph file
	*---------------------------------------------------------------------*/
	for (xadj[0] = 0, k = 0, i = 0; i < graph->nvtxs; i++) 
	{
		do {
			if (check_getline(&line, &lnlen, fpin) == -1) 
			{
				char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "Premature end of input file while reading vertex %"PRIDX".", i + 1);
				error_exit(error_message);
			}
		} while (line[0] == '%');

		curstr = line;
		newstr = NULL;

    	/* Read vertex sizes */
		if (readvs) 
		{
			vsize[i] = strtoll(curstr, &newstr, 10);
			if (newstr == curstr)
			{
				char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "The line for vertex %"PRIDX" does not have vsize information.", i + 1);
				error_exit(error_message);
			}
			if (vsize[i] < 0)
			{
				char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "The size for vertex %"PRIDX" must be >= 0.", i + 1);
				error_exit(error_message);
			}

			curstr = newstr;
		}


		/* Read vertex weights */
		if (readvw) 
		{
			for (l=0; l<1; l++) 
			{
				vwgt[i+l] = strtoll(curstr, &newstr, 10);
				if (newstr == curstr)
				{
					char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
					sprintf(error_message, "The line for vertex %"PRIDX" does not have enough weights for the %"PRIDX" constraint.", i + 1, l);
					error_exit(error_message);
				}
				if (vwgt[i + l] < 0)
				{
					char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
					sprintf(error_message, "The weight vertex %"PRIDX" and constraint %"PRIDX" must be >= 0.", i + 1, l);
					error_exit(error_message);
				}

				curstr = newstr;
			}
		}

		while (1) 
		{
			edge = strtoll(curstr, &newstr, 10);
			if (newstr == curstr)
				break; /* End of line */
			curstr = newstr;

			if (edge < 1 || edge > graph->nvtxs)
			{
				char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "Edge %"PRIDX" for vertex %"PRIDX" is out of bounds.", edge, i + 1);
				error_exit(error_message);
			}

			ewgt = 1;
			if (readew) 
			{
				ewgt = strtoll(curstr, &newstr, 10);
				if (newstr == curstr)
				{
					char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
					sprintf(error_message, "Premature end of line for vertex %"PRIDX".", i + 1);
					error_exit(error_message);
				}
				if (ewgt <= 0)
				{
					char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
					sprintf(error_message, "The weight (%"PRIDX") for edge (%"PRIDX", %"PRIDX") must be positive.", ewgt, i + 1, edge);
					error_exit(error_message);
				}

				curstr = newstr;
			}

			if (k == graph->nedges)
			{
				char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadGraph: error_message");
				sprintf(error_message, "There are more edges in the file than the %"PRIDX" specified.", graph->nedges / 2);
				error_exit(error_message);
			}

			adjncy[k] = edge-1;
			adjwgt[k] = ewgt;
			k++;
		}

    	xadj[i + 1] = k;
	}

	check_fclose(fpin);

	if (k != graph->nedges) 
	{
		printf("------------------------------------------------------------------------------\n");
		printf("***  I detected an error in your input file  ***\n\n");
		printf("In the first line of the file, you specified that the graph contained\n"
			"%"PRIDX" edges. However, I only found %"PRIDX" edges in the file.\n", 
			graph->nedges / 2, k / 2);
		if (2 * k == graph->nedges) 
		{
			printf("\n *> I detected that you specified twice the number of edges that you have in\n");
			printf("    the file. Remember that the number of edges specified in the first line\n");
			printf("    counts each edge between vertices v and u only once.\n\n");
		}
		printf("Please specify the correct number of edges in the first line of the file.\n");
		printf("------------------------------------------------------------------------------\n");
		exit(0);
	}

	// printf("lnlen=%zu sizeof(char) * lnlen=%zu\n",lnlen, sizeof(char) * lnlen);

	if (graph->tvwgt == NULL) 
		graph->tvwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 1, "SetupGraph_tvwgt: tvwgt");
	graph->tvwgt[0]    = sum_int(graph->nvtxs, vwgt, 1);

	check_free(line, sizeof(char) * lnlen, "ReadGraph: line");
	
	return graph;
}

void ReadNum(char *filename, Hunyuan_int_t length, Hunyuan_int_t *metis_match)
{
	Hunyuan_int_t i, j, k;
	FILE *fpin;

	if (!Is_file_exists(filename)) 
    {
        // char *error_message = (char *)malloc(sizeof(char) * 128);   //!!!
        char *error_message = (char *)check_malloc(sizeof(char) * 128, "ReadNum: error_message");
		sprintf(error_message, "File %s does not exist!", filename);
        error_exit(error_message);
        // errexit("File %s does not exist!\n", params->filename);
    }

    fpin = check_fopen(filename, "r", "ReadNum: Graph");

	// 读取每个整数并存入数组
	Hunyuan_int_t number;
	// printf("length=%"PRIDX"\n", length);
    for (Hunyuan_int_t i = 0; i < length; i++) 
	{
		// printf("Reading number %"PRIDX" %s\n",i,filename);
        if (fscanf(fpin, "%"PRIDX"", &number) != 1) 
		{
            perror("Error reading integer");
            free(metis_match);
            fclose(fpin);
			exit(0);
            // return EXIT_FAILURE;
        }
		// if(i < 10)
		// 	printf("i=%"PRIDX" number=%"PRIDX"\n", i, number);
        metis_match[i] = number;
    }

    // 关闭文件
    fclose(fpin);

	// 打印数组内容，用于验证
    // for (Hunyuan_int_t i = 0; i < length; i++) 
	// {
    //     printf("%"PRIDX"\n", metis_match[i]);
    // }
    // printf("\n");

}

#endif