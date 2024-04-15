input="matrix_all_1w_177.csv"
{
  read
  i=1
  while IFS=',' read -r mid Name Name1 rows cols nonzeros a b m
  do
    echo "$Name"
    #./cg_tile_omp /home/ydc/桌面/mtx/$Name.mtx 100 16
    #./cg_tile_omp_inc /home/ydc/桌面/mtx/$Name.mtx 100 16
    #./cg_tile_omp_inc_balance_v1 /home/ydc/桌面/mtx/$Name.mtx 100 16
    #./cg_tile_omp_inc_balance_v2 /home/ydc/桌面/mtx/$Name.mtx 100 16
    #./cg_tile_mix_array_omp /home/ydc/桌面/mtx/$Name.mtx 100 16
    #./cg_tile_mix_array_omp_balance /home/ydc/桌面/mtx/$Name.mtx 100 16
    i=`expr $i + 1`
  done 
} < "$input"