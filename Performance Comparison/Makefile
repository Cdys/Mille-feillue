#The environment of NVIDIA
NVCCFLAGS=-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -Xcompiler -fpermissive
LDFLAGS=-lcudart  -lcublas -lcusparse -L/usr/local/cuda-12.0/lib64 -g 
INCLUDES = -I/usr/local/cuda-12.0/include -I/home/weifeng/cuda-samples-12.0/Common 
# The environment of AMD
CUDA_TOOLKIT := $(shell dirname $$(command -v hipcc))/..
INCLUDES_HIP     := -I$(CUDA_TOOLKIT)/include
.PHONY :NVIDIA
NVIDIA:
	nvcc Mille-feuille_CG_NVIDIA.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o Mille-feuille_CG_NVIDIA -Xcompiler -fopenmp -O3 -maxrregcount=32
	nvcc Mille-feuille_BiCGSTAB_NVIDIA.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o Mille-feuille_BiCGSTAB_NVIDIA -Xcompiler -fopenmp -O3 -maxrregcount=32
	nvcc cuSPARSE_CG.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o cuSPARSE_CG -Xcompiler -fopenmp -O3
	nvcc cuSPARSE_BiCGSTAB.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o cuSPARSE_BiCGSTAB -Xcompiler -fopenmp -O3
AMD:
	hipcc Mille-feuille_CG_AMD.cu -o Mille-feuille_CG_AMD -fopenmp -lhipblas -lhipsparse -O3
	hipcc Mille-feuille_BiCGSTAB_AMD.cu -o Mille-feuille_BiCGSTAB_AMD -fopenmp -lhipblas -lhipsparse -O3
	hipcc hipSPARSE_CG.cu $(INCLUDES_HIP) -o hipSPARSE_CG -fopenmp -O3 -w -lhipblas -lhipsparse
	hipcc hipSPARSE_BiCGSTAB.cu $(INCLUDES_HIP) -o hipSPARSE_BiCGSTAB -fopenmp -O3 -w -lhipblas -lhipsparse
NVIDIA clean:
	rm Mille-feuille_CG_NVIDIA
	rm Mille-feuille_BiCGSTAB_NVIDIA
	rm cuSPARSE_CG
	rm cuSPARSE_BiCGSTAB
AMD clean:
	rm Mille-feuille_CG_AMD
	rm Mille-feuille_BiCGSTAB_AMD
	rm hipSPARSE_CG
	rm hipSPARSE_BiCGSTAB