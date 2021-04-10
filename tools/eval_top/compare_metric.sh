# export CUDA_VISIBLE_DEVICES=1

./build/seg_eval
python deploy/bin2png.py

echo "c++ metric"
python deploy/eval_predict_imgs.py


echo "python metric:"
python val.py --config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml \
--model_path saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_bs64_lr0.1_iter2w_horizontal_distort/best_model/model.pdparams
