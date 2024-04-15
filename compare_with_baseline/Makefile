NVCCFLAGS=-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -Xcompiler -fpermissive
LDFLAGS=-lcudart  -lcublas -lcusparse -L/usr/local/cuda-12.0/lib64 -g 
INCLUDES = -I/usr/local/cuda-12.0/include -I/home/weifeng/cuda-samples-12.0/Common 
.PHONY :all
all:
	nvcc Mille-feuille_CG.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o Mille-feuille_CG -Xcompiler -fopenmp -O3 -maxrregcount=32
	nvcc Mille-feuille_BiCGSTAB.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o Mille-feuille_BiCGSTAB -Xcompiler -fopenmp -O3 -maxrregcount=32
	nvcc cuSPARSE_CG.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o cuSPARSE_CG -Xcompiler -fopenmp -O3
	nvcc cuSPARSE_BiCGSTAB.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o cuSPARSE_BiCGSTAB -Xcompiler -fopenmp -O3
clean:
	rm Mille-feuille_CG
	rm Mille-feuille_BiCGSTAB
	rm cuSPARSE_CG
	rm cuSPARSE_BiCGSTAB