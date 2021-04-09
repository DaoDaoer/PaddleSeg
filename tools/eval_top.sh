python eval_top/infer.py --config eval_top/fcn_hrnetw18_small_v1_not_fix_shape_argmax_humanseg_192x192_bs64_lr0.1_iter2w_horizontal_distort/deploy.yaml \
--file_path eval_top/mini_supervisely/val.txt \
--data_dir eval_top/mini_supervisely \
--save_dir eval_top/python_predict \
--without_argmax
