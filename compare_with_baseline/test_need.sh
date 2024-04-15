input="cg_matrix_set.csv"
{
  read
  i=1
  while IFS=',' read -r data a b c d
  do
    for data1 in `find "/home/weifeng/MM/" -name "$data.mtx"`
    do
        echo -n $data1
        #timeout -s 9 1m ./bykrylov_bicg_syncfree $data1 16
        #timeout -s 9 1m ./bykrylov_bicg_syncfree_resort $data1 $each_nnz
        timeout -s 9 1m ./bykrylov_initial $data1
        #timeout -s 9 1m ./bykrylov_bicg_syncfree_resort_mixed_precision $data1 $each_nnz
        # timeout -s 9 1m ./bykrylov_cg_syncfree_resort_mixed_precision $data1 32
        # timeout -s 9 1m ./bykrylov_cg_syncfree_resort_mixed_precision $data1 64
        # timeout -s 9 1m ./bykrylov_cg_syncfree_resort_mixed_precision $data1 128
        # timeout -s 9 1m ./bykrylov_cg_syncfree_resort_mixed_precision $data1 256
        # timeout -s 9 1m ./bykrylov_cg_syncfree_resort_mixed_precision $data1 512
        # timeout -s 9 1m ./bykrylov_cg_syncfree_resort_mixed_precision $data1 1024
    done
    i=`expr $i + 1`
  done 
} < "$input"