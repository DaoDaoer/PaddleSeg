// #include "opencv2/core.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/imgproc.hpp"
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
#include "glog/logging.h"
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


  ///////////////////////////////////// config
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

  if (input_data_path.empty()) {
    cout << "FLAGS_input_data_path is empty.";
  }
  std::ifstream input_fs(input_data_path, std::ios::binary);
  if (!input_fs.is_open()) {
      cout << "open input image " << input_data_path << " error.";
  }
  int64_t img_nums = 0;
  int batch_size = 1;
  input_fs.read((char*)&img_nums, sizeof(img_nums));
  std::ofstream output_fs("predict.bin", std::ios::binary);

  Timer infer_time;

  for (int img_idx = 0 ; img_idx < img_nums ; img_idx++){
      int img_size = 1 * channel * height * width;
      std::vector<float> input_data(img_size, 0.0f);

      input_fs.read((char*)&input_data[0], sizeof(float)*img_size);

      // Inference.
      infer_time.start()
      auto input_names = predictor->GetInputNames();
      auto input_t = predictor->GetInputHandle(input_names[0]);
      input_t->Reshape({1, channel, height, width});
      input_t->CopyFromCpu(input_data.data());

      predictor->Run();

      std::vector<int64_t> out_data;
      auto output_names = predictor->GetOutputNames();
      auto output_t = predictor->GetOutputHandle(output_names[0]);
      std::vector<int> output_shape = output_t->shape();
      int out_size = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                    std::multiplies<int>());

      out_data.resize(out_size);
      output_t->CopyToCpu(out_data.data());
      infer_time.stop()
      ////////////////////////////////////////

      output_fs.write((char*)&out_data[0], sizeof(int64_t) * out_size);


  LOG(INFO) << "----------------------- Model info ----------------------";
  LOG(INFO) << "Model name: " << "fcn_hrnetw18_small_v1" ;
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "Batch size: " << batch_size << ", " \
              "Num of samples: " << img_nums;
  LOG(INFO) << "input_shape: "
          << "dynamic shape";
  LOG(INFO) << "----------------------- Conf info -----------------------";
    LOG(INFO) << "device: " << (config.use_gpu() ? "gpu" : "cpu") << ", " \
                "ir_optim: " << (config.ir_optim() ? "true" : "false");
    LOG(INFO) << "enable_memory_optim: " << (config.enable_memory_optim() ? "true" : "false");
    if (config.use_gpu()) {
      LOG(INFO) << "enable_tensorrt: " << (config.tensorrt_engine_enabled() ? "true" : "false");
      if (config.tensorrt_engine_enabled()) {
        LOG(INFO) << "precision: " << (config.use_fp16 ? "fp16" : "fp32");
      }
    }else {
      LOG(INFO) << "enable_mkldnn: " << (config.mkldnn_enabled() ? "true" : "false");
      LOG(INFO) << "cpu_math_library_num_threads: " << config.cpu_math_library_num_threads();
    }
  LOG(INFO) << "----------------------- Perf info -----------------------";
  LOG(INFO) << "Average latency(ms): " << infer_time.report() / img_nums;

  }
  return 0;
}


class Timer {
// Timer, count in ms
  public:
      Timer() {
          reset();
      }
      void start() {
          start_t = std::chrono::high_resolution_clock::now();
      }
      void stop() {
          auto end_t = std::chrono::high_resolution_clock::now();
          typedef std::chrono::microseconds ms;
          auto diff = end_t - start_t;
          ms counter = std::chrono::duration_cast<ms>(diff);
          total_time += counter.count();
      }
      void reset() {
          total_time = 0.;
      }
      double report() {
          return total_time / 1000.0;
      }
  private:
      double total_time;
      std::chrono::high_resolution_clock::time_point start_t;
};
