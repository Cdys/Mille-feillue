# the output will be
# program, matirx, bytes of A & x, bytes of y, bytes of computation,
#              matrix columns, matrix rows, nnzs, execution time(ms)

check(){
    file_path=$1
    mat_name=${file_path##*/}
    mat_name=${mat_name%.*}
    parent_name=${file_path%/*}
    parent_name=${parent_name##*/}
    #echo $parent_name
    #echo $mat_name
    FILE=$1
    FILE_SIZE=`du $FILE | awk '{print $1}'`
    if [ $FILE_SIZE -ge $((1024*1024*15)) ]
    then
        return 1
    fi

    if [ "$parent_name" = "$mat_name" ]; then
        return 0 # 0 eq true
    else
        return 1
    fi
}

test_all(){
    for data in `find $2 -name "*.mtx"`
    do
        if check $data; then
            echo -n $data
            echo -n ","
            #timeout -s 9 10m ./test_ex1 $data -ksp_max_it 1000 -ksp_monitor -ksp_type bicg -mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0 -pc_type none -ksp_norm_type unpreconditioned
            #timeout -s 9 10m ./test_ex1 $data -ksp_max_it 1000 -ksp_monitor -ksp_type cg -mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0 -pc_type none -ksp_norm_type unpreconditioned
            timeout -s 9 10m  ./test_ex1 $data -ksp_max_it 1000 -ksp_type gmres -ksp_gmres_restart 1 -mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0 -pc_type none -ksp_norm_type unpreconditioned
            #timeout -s 9 20m ./test_ex1 $data -ksp_max_it 500 -ksp_monitor -ksp_type gmres -ksp_gmres_restart 1 
        fi
    done
}

exe_path="./"
#data_path="/home/weifeng/MM/Engwirda/airfoil_2d/"
#data_path="/ssget/MM"
data_path="/home/weifeng/MM"

test_all $exe_path $data_path
