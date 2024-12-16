all:
	# nvcc -std=c++11 -gencode arch=compute_89,code=sm_89 -O3  hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w
	nvcc -std=c++11 -gencode arch=compute_86,code=sm_86 -O3  hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w -g
	# nvcc -std=c++11 -gencode arch=compute_86,code=sm_86 -O3  hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w
	# nvcc -std=c++11 -gencode arch=compute_89,code=sm_89 -O3  cuMetis.cu -o  cuMetis  --expt-relaxed-constexpr -w -g -G
