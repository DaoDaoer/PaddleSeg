export CUDA_VISIBLE_DEVICES=5

model_name="fcn_hrnetw18_small_v1"
batch_size=1
device="cpu"
trt="false"
mkldnn="false"
cpu_math_library_num_threads=6

mkdir -p log

for device in "gpu" "cpu"
do
    if [ $device == "cpu" ]
    then
        for mkldnn in "false" "true"
        do

            python infer.py --config fcn_hrnetw18_small_v1_not_fix_shape_argmax_humanseg_192x192_bs64_lr0.1_iter2w_horizontal_distort/deploy.yaml \
            --file_path mini_supervisely/val.txt \
            --data_dir mini_supervisely \
            --save_dir python_predict \
            --without_argmax \
            --batch_size ${batch_size} \
            --device ${device} \
            --enable_mkldnn ${mkldnn} | tee log/${model_name}_bs${batch_size}_${device}_trt_${trt}_mkldnn_${mkldnn}.log

            ./build/seg_eval --use_gpu false --mkldnn $mkldnn --cpu_math_library_num_threads $cpu_math_library_num_threads
        done

    else
        for trt in "false" "true"
        do
            echo $trt
            python infer.py --config fcn_hrnetw18_small_v1_not_fix_shape_argmax_humanseg_192x192_bs64_lr0.1_iter2w_horizontal_distort/deploy.yaml \
            --file_path mini_supervisely/val.txt \
            --data_dir mini_supervisely \
            --save_dir python_predict \
            --without_argmax \
            --batch_size ${batch_size} \
            --device ${device} \
            --use_trt ${trt} | tee log/${model_name}_bs${batch_size}_${device}_trt_${trt}_mkldnn_${mkldnn}.log
        done
    fi
done



use_gpu=0
use_mkldnn=0
cpu_math_library_num_threads=6
use_tensorrt=0

for use_gpu in 0 1
do
    if [$use_gpu == 0]
    then
        for use_mkldnn in 0 1
        do
            for cpu_math_library_num_threads in 1 6
            do
                ./build/seg_eval --use_gpu $use_gpu --use_mkldnn $use_mkldnn --cpu_math_library_num_threads $cpu_math_library_num_threads \
                | tee log/${model_name}_bs${batch_size}_use_gpu${use_gpu}_use_trt${use_trt}_use_mkldnn${use_mkldnn}_cpu_threads${cpu_math_library_num_threads}.log
            done
    else
        for use_tensorrt in 0 1
        do
            ./build/seg_eval --use_gpu $use_gpu --use_tensorrt $use_tensorrt \
            | tee log/${model_name}_bs${batch_size}_use_gpu${use_gpu}_use_trt${use_trt}_use_mkldnn${use_mkldnn}_cpu_threads${cpu_math_library_num_threads}.log
        done
    fi
done
