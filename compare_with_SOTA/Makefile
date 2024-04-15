#The environment of NVIDIA
NVCCFLAGS=-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -Xcompiler -fpermissive
LDFLAGS=-lcudart  -lcublas -lcusparse -L/usr/local/cuda-12.0/lib64 -g 
INCLUDES = -I/usr/local/cuda-12.0/include -I/home/weifeng/cuda-samples-12.0/Common 
.PHONY :NVIDIA
NVIDIA:
	nvcc Mille-feuille_CG_NVIDIA.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o Mille-feuille_CG_NVIDIA -Xcompiler -fopenmp -O3 -maxrregcount=32
	nvcc Mille-feuille_BiCGSTAB_NVIDIA.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o Mille-feuille_BiCGSTAB_NVIDIA -Xcompiler -fopenmp -O3 -maxrregcount=32
NVIDIA clean:
	rm Mille-feuille_CG_NVIDIA
	rm Mille-feuille_BiCGSTAB_NVIDIA