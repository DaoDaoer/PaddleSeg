#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>
#include <fstream>
#include <include/cmdline.h>
#include <dirent.h>
#include<sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <numeric>

using namespace paddle_infer;
using namespace std;


int main(int argc, char **argv) {

  cmdline::parser parser;
  parser.add<std::string>("input_data_path",'\0',"",false,"./data/val_imgs");
  parser.add<std::string>("model_dir",'\0',"",false,"./model/fcn_hrnetw18_small_v1_not_fix_shape_argmax_humanseg_192x192_bs64_lr0.1_iter2w_horizontal_distort/");
  parser.add<int>("channel",'\0',"",false,3);
  parser.add<int>("height",'\0',"",false,192);
  parser.add<int>("width",'\0',"",false,192);

  parser.add<bool>("use_gpu",'\0',"",false,true);
  parser.add<bool>("use_mkldnn",'\0',"",false,false);
  parser.add<bool>("use_tensorrt",'\0',"",false,false);
  parser.add<bool>("use_fp16",'\0',"",false,false);
  parser.add<int>("gpu_mem",'\0',"",false,4000);
  parser.add<int>("gpu_id",'\0',"",false,0);
  parser.add<int>("cpu_math_library_num_threads",'\0',"",false,6);
  parser.parse_check(argc, argv);

  std::string input_data_path = parser.get<std::string>("input_data_path");
  std::string model_dir = parser.get<std::string>("model_dir");
  int channel = parser.get<int>("channel");
  int height = parser.get<int>("height");
  int width = parser.get<int>("width");

  bool use_gpu = parser.get<bool>("use_gpu");
  bool use_mkldnn = parser.get<bool>("use_mkldnn");
  bool use_tensorrt = parser.get<bool>("use_tensorrt");
  bool use_fp16 = parser.get<bool>("use_fp16");

  int gpu_mem = parser.get<int>("gpu_mem");
  int gpu_id = parser.get<int>("gpu_id");
  int cpu_math_library_num_threads = parser.get<int>("cpu_math_library_num_threads");


  /////////////////////////////////////
  /////////////////////////////////////
  /////////////////////////////////////
  paddle_infer::Config config;
  config.SetModel(model_dir + "model.pdmodel",
                  model_dir + "model.pdiparams");

  if (use_gpu) {
    config.EnableUseGpu(gpu_mem, gpu_id);
    if (use_tensorrt) {
      config.EnableTensorRtEngine(
          1 << 20, 10, 3,
          use_fp16 ? paddle_infer::Config::Precision::kHalf
                          : paddle_infer::Config::Precision::kFloat32, false, false);
    }
  } else {
    config.DisableGpu();
    if (use_mkldnn) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(cpu_math_library_num_threads);
  }

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  //config.DisableGlogInfo();

  std::shared_ptr<Predictor> predictor = CreatePredictor(config);
  /////////////////////////////////////
  /////////////////////////////////////
  /////////////////////////////////////


  // std::vector<std::string> all_inputs;

  // showAllFiles(input_data_path.c_str(), all_inputs);
  if (input_data_path.empty()) {
    cout << "FLAGS_input_data_path is empty.";
  }
  std::ifstream input_fs(input_data_path, std::ios::binary);
  if (!input_fs.is_open()) {
      cout << "open input image " << input_data_path << " error.";
  }

  int64_t img_nums = 0;
  input_fs.read((char*)&img_nums, sizeof(img_nums));

  std::ofstream output_fs("predict.bin", std::ios::binary);

  for (int img_idx = 0 ; img_idx < img_nums ; img_idx++){
      int img_size = 1 * channel * height * width;
      std::vector<float> input_data(img_size, 0.0f);
      input_fs.read((char*)&input_data, sizeof(float)*img_size);


      // Inference.
      auto input_names = predictor->GetInputNames();
      auto input_t = predictor->GetInputHandle(input_names[0]);
      input_t->Reshape({1, channel, height, width});
      input_t->CopyFromCpu(input_data.data());

      predictor->Run();

      std::vector<float> out_data;
      auto output_names = predictor->GetOutputNames();
      auto output_t = predictor->GetOutputHandle(output_names[0]);
      std::vector<int> output_shape = output_t->shape();
      int out_size = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                    std::multiplies<int>());

      out_data.resize(out_size);
      output_t->CopyToCpu(out_data.data());
      ////////////////////////////////////////

      output_fs.write((char*)&out_data, sizeof(int64_t) * out_size);

      // ofstream out_file(save_path + all_inputs[img_idx]);

      // for (int idx_h = 0; idx_h < height; idx_h++){
      //   for (int idx_w = 0; idx_w < width; idx_w++){

      //     out_file << ("%.6f", out_data[idx_h * width + idx_w]);

      //     if (idx_w == width - 1){
      //       out_file << '\n';
      //     }else{
      //       out_file << ' ';
      //     }
      //   }
      // }

      // //cout << img_idx <<"  /  " << img_nums << endl;
  }
  return 0;
}
