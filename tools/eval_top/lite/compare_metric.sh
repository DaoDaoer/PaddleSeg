export CUDA_VISIBLE_DEVICES=5




# # infernce python
# python3.7 infer.py --config model/fcn_hrnetw18_small_v1_not_fix_shape_argmax_humanseg_192x192_bs64_lr0.1_iter2w_horizontal_distort/deploy.yaml \
# --file_path data/mini_supervisely/val.txt \
# --data_dir data/mini_supervisely \
# --save_dir predict_python/Images \
# --without_argmax
echo "*************************\n" | tee  log.txt
echo "*************************\n" | tee -a log.txt
echo "******** python:  *******\n" | tee -a log.txt
echo "*************************\n" | tee -a log.txt
echo "*************************\n" | tee -a log.txt
python3.7 eval_predict_imgs.py --pred_dir predict_python/ | tee -a log.txt



# lite cxx
python3.7 bin2png.py --bin_file_path lite/armv7/predict.bin --save_dir lite/armv7/predict_cxx/

echo "*************************\n" | tee lite/armv7/log.txt
echo "*************************\n" | tee -a lite/armv7/log.txt
echo "******** c++:  **********\n" | tee -a lite/armv7/log.txt
echo "*************************\n" | tee -a lite/armv7/log.txt
echo "*************************\n" | tee -a lite/armv7/log.txt
python3.7 eval_predict_imgs.py --pred_dir lite/armv7/predict_cxx/ | tee -a lite/armv7/log.txt
