input="cg_matrix_set.csv" #The name of dataset
{
  read
  i=1
  while IFS=',' read -r name nnz
  do
    for matrix in `find "/home/weifeng/MM/" -name "$name.mtx"` #The road of data
    do
        ./Mille-feuille_CG_NVIDIA $matrix
        ./Mille-feuille_BiCGSTAB_NVIDIA $matrix
        ./cuSPARSE_CG $matrix
        ./cuSPARSE_BiCGSTAB $matrix
    done
    i=`expr $i + 1`
  done 
} < "$input"