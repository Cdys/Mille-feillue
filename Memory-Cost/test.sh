input="CG_dataset" #The name of dataset
{
  read
  i=1
  while IFS=',' read -r name nnz
  do
    for matrix in `find "/home/dataset/MM/" -name "$name.mtx"` #The road of data
    do
        ./memory_Mille_feillue $matrix
        ./memory_cuSPARSE $matrix
    done
    i=`expr $i + 1`
  done 
} < "$input"