#The environment of NVIDIA
NVCCFLAGS=-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -Xcompiler -fpermissive
LDFLAGS=-lcudart  -lcublas -lcusparse -L/usr/local/cuda-12.0/lib64 -g 
INCLUDES = -I/usr/local/cuda-12.0/include -I/home/weifeng/cuda-samples-12.0/Common 
.PHONY :NVIDIA
NVIDIA:
	nvcc memory_Mille_feillue.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o memory_Mille_feillue -Xcompiler -fopenmp -O3 -maxrregcount=32
	nvcc memory_cuSPARSE.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o memory_cuSPARSE -Xcompiler -fopenmp -O3
NVIDIA clean:
	rm memory_Mille_feillue
	rm memory_cuSPARSE