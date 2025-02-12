# Note for running Deepseek R1 and R1-Qwen-32B


### 有用的链接
+ R1 全量模型下载链接： `https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d`
+ R1 量化版本 - unsloth版，推荐3bit以内用这个: `https://huggingface.co/unsloth/DeepSeek-R1-GGUF`
+ R1 量化版本 - bartowski版，各个bit的都有: `https://huggingface.co/bartowski/DeepSeek-R1-GGUF`
+ R1-Qwen-32B全量版本：`https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/tree/main`
+ R1-Qwen-32B量化版本，目前bartowski的下载比unsloth多：`https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF`
+ 4x4090部署R1 量化版本教程：`https://snowkylin.github.io/blogs/a-note-on-deepseek-r1.html`
+ unsloth的量化版本教程：`https://unsloth.ai/blog/deepseekr1-dynamic`
+ Open-WebUI跑R1量化教程: `https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/`
+ 怎么选GGUF量化模型？`https://huggingface.co/bartowski/DeepSeek-R1-GGUF#which-file-should-i-choose`
+ 怎么选量化模型2: `https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9`
+ llama.cpp server调参教程: `https://blog.steelph0enix.dev/posts/llama-cpp-guide/#llamacpp-server-settings`
+ 模型量化tutorial: `https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization`
+ 把本地机器通过tunnel让外界访问web demo: `https://medium.com/design-bootcamp/how-to-setup-a-cloudflare-tunnel-and-expose-your-local-service-or-application-497f9cead2d3`

### 部署教程

测试环境包括：
+ 8卡A6000，共384GB显存VRAM
+ 4卡3090=96GB,  单卡3090=24GB
+ 4卡L40=~180GB
+ ARM系列，Orin NX 16 GB, Orin AGX 64 GB

#### 速度测试总结

+ 以下包含了30万元预算(8卡A6000)、20万元预算(4卡L40)、5万元预算(4卡3090)、2万元预算电脑(1卡3090)跑Deepseek速度测试，只测了8k短序列
+ 全部塞GPU; 30 token/s 以上用的才比较舒服，类似chatGPT体感

+ 精华表

| 模型                                       | 模型大小  | 测试机器               | 工具        | Note                          | 使用显存    | tg速度throughput |
| ---------------------------------------- | ----- | ------------------ | --------- | ----------------------------- | ------- | -------------- |
| DeepSeek-R1-Distill-Qwen-32B             | 62GB  | 4x3090, 64核256GB   | SGLang    | 不支持FP8推理; *5万元电脑               | 22GB x4 | 40 token/s     |
|                                          |       | 8xA6000, 128核512GB | SGLang    | \*30万元电脑                      | 46GB x8 | 55 token/s     |
|                                          |       | 4xL40, 128核512GB   | SGLang    | \--quantization fp8; \*20万元电脑 | 43GB x4 | 55 token/s     |
| DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf | 19GB  | 4x3090, 64核256GB   | llama.cpp | \*2万元电脑                       | 21GB x1 | 26 token/s     |
|                                          |       | AGX Orin 64GB      | llama.cpp | 30W/50W; \*2万元边缘设备            | 20.2G   | 5 token/s      |
|                                          |       | AGX Orin 64GB      | llama.cpp | MAX (60W)                     | 20.2G   | 6.8 token/s    |
| DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf | 9GB   | AGX Orin 64GB      | llama.cpp | MAX                           | 11.5G   | 14 token/s     |
| DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf   | 11GB  | AGX Orin 64GB      | llama.cpp | MAX                           | 14.8G   | 12 token/s     |
| DeepSeek-R1-UD-IQ1_M                     | 158GB | 8xA6000, 128核512GB | llama.cpp |                               | 26GB x8 | 18 token/s     |
| DeepSeek-R1-UD-IQ1_S                     | 131GB | 4xL40, 128核512GB   | llama.cpp | \-c 4096否则OOM                 | 39GB x4 | 23 token/s     |
| DeepSeek-R1-IQ4_XS                       | 333GB | 8xA6000, 128核512GB | llama.cpp | \-c 1024否则OOM                 | 47GB x8 | 15 token/s     |

+ 全部测试

| 模型                                       | 模型大小  | 测试机器               | 工具        | Note                          | 使用显存    | tg速度throughput |
| ---------------------------------------- | ----- | ------------------ | --------- | ----------------------------- | ------- | -------------- |
| DeepSeek-R1-Distill-Qwen-32B             | 62GB  | 4x3090, 64核256GB   | SGLang    | 不支持FP8推理; 5万元电脑               | 22GB x4 | 40 token/s     |
| DeepSeek-R1-Distill-Qwen-32B             | 62GB  | 8xA6000, 128核512GB | SGLang    |                               | 47GB x4 | 36 token/s     |
|                                          |       | 8xA6000, 128核512GB | SGLang    | \*30万元电脑                      | 46GB x8 | 55 token/s     |
| DeepSeek-R1-Distill-Qwen-32B             | 62GB  | 4xL40, 128核512GB   | SGLang    |                               | 43GB x4 | 41 token/s     |
|                                          |       | 4xL40, 128核512GB   | SGLang    | \--quantization fp8; \*20万元电脑 | 43GB x4 | 55 token/s     |
| DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf | 19GB  | 8xA6000, 128核512GB | llama.cpp |                               | 4GB x8  | 27 token/s     |
|                                          |       | 8xA6000, 128核512GB | llama.cpp |                               | 6GB x4  | 28 token/s     |
|                                          |       | 8xA6000, 128核512GB | llama.cpp |                               | 21GB x1 | 29 token/s     |
|                                          |       | 4xL40, 128核512GB   | llama.cpp |                               | 6GB x4  | 30 token/s     |
|                                          |       | 4xL40, 128核512GB   | llama.cpp |                               | 21GB x1 | 31 token/s     |
|                                          |       | 4x3090, 64核256GB   | llama.cpp |                               | 6GB x4  | 32 token/s     |
|                                          |       | 4x3090, 64核256GB   | llama.cpp | \*2万元电脑                       | 21GB x1 | 26 token/s     |
|                                          |       | 4x3090, 64核256GB   | SGLang    | 这个很奇怪，有可能胡言乱语                 | 22GB x4 | 67 token/s     |
| DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf | 19GB  | AGX Orin 64GB      | llama.cpp | 30W/50W                       | 20.2G   | 5 token/s      |
|                                          |       | AGX Orin 64GB      | llama.cpp | MAX                           | 20.2G   | 6.8 token/s    |
| DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf | 9GB   | AGX Orin 64GB      | llama.cpp | MAX; \*2万元边缘设备                | 11.5G   | 14 token/s     |
| DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf   | 11GB  | AGX Orin 64GB      | llama.cpp | MAX                           | 14.8G   | 12 token/s     |
| DeepSeek-R1-UD-IQ1_M                     | 158GB | 4xL40, 128核512GB   | llama.cpp | 4卡OOM                         |         |                |
| DeepSeek-R1-UD-IQ1_S                     | 131GB | 4xL40, 128核512GB   | llama.cpp | \-c 4096否则OOM                 | 39GB x4 | 23 token/s     |
| DeepSeek-R1-UD-IQ1_M                     | 158GB | 4x3090, 64核256GB   | llama.cpp |                               | 21GB x4 | 3.2 token/s    |
| DeepSeek-R1-UD-IQ1_M                     | 158GB | 8xA6000, 128核512GB | llama.cpp | 4卡OOM                         | 26GB x8 | 18 token/s     |
| DeepSeek-R1-IQ4_XS                       | 333GB | 8xA6000, 128核512GB | llama.cpp | \-c 1024否则OOM                 | 47GB x8 | 15 token/s     |
| DeepSeek-R1-Q4_K_M                       | 377GB | 8xA6000, 128核512GB | llama.cpp | 8卡OOM                         |         |                |
| DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf   | 33GB  | 4x3090, 64核256GB   | llama.cpp |                               | 10GB x4 | 21 token/s     |

#### R1-Qwen-32B

+ 下载模型
```
    # 安装huggingface 命令行下载工具，运行同样的命令可以自动断点续传
        # 可能要挂了VPN才能下载，否则报domain name错误

            $ pip install -U "huggingface_hub[cli]"

        # 可以尝试使用huggingface 镜像网站
            $ export HF_ENDPOINT=https://hf-mirror.com
            $ huggingface-cli download ...

    # 全量: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/tree/main

        (base) junweil@home-lab:/mnt/nvme2/junweil/deepseek$ huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir ./

    # 量化版本: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF
        # Q8和Q4版本

            (base) junweil@home-lab:/mnt/nvme2/junweil/deepseek$ huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF --include "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf" --local-dir ./

            (base) junweil@home-lab:/mnt/nvme2/junweil/deepseek$ huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF --include "DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf" --local-dir ./
```

+ 用SGLang跑全量R1-Qwen-32B模型 (FP16 inference, 60多GB模型文件，需要60多GB显存)
```
    # 网上说SGLang比vLLM更快，Deepseek团队也首推SGLang: https://medium.com/@zhaochenyang20/%E5%B0%8F%E7%99%BD%E8%A7%86%E8%A7%92-vllm-%E8%BF%81%E7%A7%BB%E5%88%B0-sglang-%E7%9A%84%E4%BD%93%E9%AA%8C%E4%B8%8E%E6%94%B6%E8%8E%B7-ca71cd55982b
        # SGLang支持Open AI 同款server API，还支持Qwen-VL: https://docs.sglang.ai/backend/openai_api_vision.html

    # https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#6-how-to-run-locally

    1. 安装SGLang
        $ conda create -n deepseek python=3.10
        $ conda activate deepseek
            # 清华源： pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
        #  https://docs.sglang.ai/start/install.html
        # got
            sglang-0.4.2.post1
            flashinfer-0.1.6+cu124torch2.4
            and a lot more

                numpy-1.26.4 # so many warning with scipy, A NumPy version >=1.17.3 and <1.25.0
                # so pip install numpy==1.24
                # numba SystemError: initialization of _internal failed without raising an exception
                    Update numba-0.61.0 solved

            # SGLang launch documentation: https://docs.sglang.ai/backend/server_arguments.html

                --dp 2 : for data parallelism
                --tp 2 : for tensor/model parallelism

                # more hyperparam tuning guide: https://docs.sglang.ai/references/hyperparameter_tuning.html

        # on machine5, 4x3090, no NVLink so --enable-p2p-check

            --dp 2 will has CUDA out of memory, --tp 2 will also CUDA out of memory, --tp 4 works

            (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ python -m sglang.launch_server --model-path models/DeepSeek-R1-Distill-Qwen-32B --tp 4 --enable-p2p-check --host 0.0.0.0 --port 6666 --trust-remote-code

                max_total_num_tokens=67772, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=2049, context_len=131072

                # all 4 gpu each got 22.86GB/24GB; model disk size is 62 GB

                # --quantization fp8 does not work
                    # RuntimeError: torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+

                    $ nvidia-smi --query-gpu=compute_cap --format=csv
                    $ 3090 is 8.6

                    # need Ada Lovelace (4090, H100) L40?

                # --enable-torch-compile: this takes 7 minutes to load up the server
                    # 每次启动都需要等

                # 如果遇到 PermissionError: [Errno 13] Permission denied: ''
                    # 需要重新关闭screen，开一个新的screen再跑

        # on machine4
            # with sglang-0.4.2.post2

            # need --mem-fraction-static 0.8 (default 0.9) to avoid OOM since desktop display takes some GPU memory

            (deepseek) junweil@ai-precog-machine4:/mnt/ssd1/junweil/deepseek$ python -m sglang.launch_server --model-path models/DeepSeek-R1-Distill-Qwen-32B --tp 4 --enable-p2p-check --host 0.0.0.0 --port 6666 --trust-remote-code --enable-torch-compile --mem-fraction-static 0.8

            # 41 token/s

        # 安装前端，我们使用Open WebUI，不然自己写程序测试也可以
            # Qwen2.5+vLLM+Open-WebUI 教程： https://jklincn.com/posts/qwen-vllm-deploy/

            # 需要起一个新环境，一个新的screen，Open-WebUI需要python3.11
                $ conda create -n openweb python=3.11
                $ conda activate openweb
                $ python3 -m pip install open-webui

                (openweb) junweil@ai-precog-machine5:~$ export ENABLE_OLLAMA_API=False
                (openweb) junweil@ai-precog-machine5:~$ export OPENAI_API_BASE_URL=http://127.0.0.1:6666/v1
                (openweb) junweil@ai-precog-machine5:~$ open-webui serve

                    # first-time it will download some example models?

                # 然后打开浏览器上 http://10.7.9.156:8080/ 就可以使用了

                # 运行体验：
                    [2025-02-03 13:17:56 TP0] Decode batch. #running-req: 1, #token: 2179, token usage: 0.03, gen throughput (token/s): 36.82, #queue-req: 0
                        # see what these mean: https://docs.sglang.ai/references/hyperparameter_tuning.html#tune-your-request-submission-speed

                    # --tp 4 基本10秒内出结果，这里是说 37 token/s? 界面可以点开思考过程，有些时候tag不对就散开了。


                    # --enable-torch-compile: 似乎快一些，throughput 37 -> 40
                        [2025-02-03 13:44:11 TP0] Decode batch. #running-req: 1, #token: 593, token usage: 0.01, gen throughput (token/s): 40.42, #queue-req: 0


```

+ 用SGLang跑量化模型
```
    # DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf
        (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek/models$ python -m sglang.launch_server --model-path DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --tp 4 --enable-p2p-check --host 0.0.0.0 --port 6666 --trust-remote-code --enable-torch-compile --mem-fraction-static 0.8 --quantization gguf

            # --mem-fraction-static 0.9: OOM
            # 占用显存： 22GB x4 ??

            # 67 token/s

            # 但是输出会有问题！可能会胡言乱语了

    # DeepSeek-R1-UD-IQ1_M does not work yet
        (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek/models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M$ python -m sglang.launch_server --model-path DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf --tp 4 --enable-p2p-check --host 0.0.0.0 --port 6666
        --trust-remote-code --mem-fraction-static 0.6 --quantization gguf

    # 感觉跑量化模型，还是用llama.cpp更好

```

+ 用llama.cpp跑量化模型 (ollama应该安装容易很多，但是也需要llama.cpp去把GGUF文件合并成一个)
```
    1. 安装
        # 直接下载llama.cpp binaries，也可以自己build
            # https://github.com/ggerganov/llama.cpp/releases
        # build 教程：https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device#using-llama.cpp-recommended

        $ git clone https://github.com/ggerganov/llama.cpp

        # need CUDA build kit
        $ sudo apt install nvidia-cuda-toolkit

            # g++ and CUDA error: /usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:

            # 这个错误是因为CUDA版本问题，去官网下载CUDA 12.4, 然后nvcc --version看是不是12.4，然后用g++-11成功编译

            (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ cmake llama.cpp -B llama.cpp/build     -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON

            (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ cmake --build llama.cpp/build --config Release -j 8 --clean-first

            (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ cp llama.cpp/build/bin/llama-* llama.cpp

            # 现在相关的binaries都在llama.cpp文件夹下了

    2. Serve GGUF model!
        # all the parameters: https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
            # a guide: https://blog.steelph0enix.dev/posts/llama-cpp-guide/#llamacpp-server-settings

        # DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf, 19 GB model

            # 先跑一次下面命令，需要确定有多少layer可以放GPU

            (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ ./llama.cpp/llama-server --model models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --port 8888 --ctx-size 8192 --n-gpu-layers -1

            load_model: loading model 'models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf'
            llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 3090) - 23982 MiB free
            llama_model_load_from_file_impl: using device CUDA1 (NVIDIA GeForce RTX 3090) - 23982 MiB free
            llama_model_load_from_file_impl: using device CUDA2 (NVIDIA GeForce RTX 3090) - 23982 MiB free
            llama_model_load_from_file_impl: using device CUDA3 (NVIDIA GeForce RTX 3090) - 23923 MiB free

            print_info: file format = GGUF V3 (latest)
            print_info: file type   = Q4_K - Medium
            print_info: file size   = 18.48 GiB (4.85 BPW)
                    print_info: arch             = qwen2
            print_info: vocab_only       = 0
            print_info: n_ctx_train      = 131072
            print_info: n_embd           = 5120
            print_info: n_layer          = 64

            load_tensors: offloading 0 repeating layers to GPU
            load_tensors: offloaded 0/65 layers to GPU
            load_tensors:   CPU_Mapped model buffer size = 18926.01 MiB

            # 放完全部，那就写65

            (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ ./llama.cpp/llama-server --model models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --port 8888 --ctx-size 8192 --n-gpu-layers 65


            load_tensors: offloading 64 repeating layers to GPU
            load_tensors: offloading output layer to GPU
            load_tensors: offloaded 65/65 layers to GPU
            load_tensors:   CPU_Mapped model buffer size =   417.66 MiB
            load_tensors:        CUDA0 model buffer size =  4844.72 MiB
            load_tensors:        CUDA1 model buffer size =  4366.53 MiB
            load_tensors:        CUDA2 model buffer size =  4366.53 MiB
            load_tensors:        CUDA3 model buffer size =  4930.57 MiB

            # 然后起一下web UI
                (openweb) junweil@ai-precog-machine5:~$ export OPENAI_API_BASE_URL=http://127.0.0.1:8888/v1
                (openweb) junweil@ai-precog-machine5:~$ open-webui serve

            # Used 6GB for each of the 4 GPUs, 20-30% utilization


               prompt eval time =    2758.37 ms /  2394 tokens (    1.15 ms per token,   867.90 tokens per second)
               eval time =    7359.65 ms /   229 tokens (   32.14 ms per token,    31.12 tokens per second)
              total time =   10118.02 ms /  2623 tokens

            # 感受: 还行，10秒出结果，32 token/s。第一个问题要30秒预热。回答效果和全量模型没看出区别



            --split-mode row : 明显慢，18 token/s，占用显存一样
            --split-mode none: 放一张卡上，21 GB显存占用, 14秒延迟，26 token/s

            --prio 3: set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: 0)
                # 似乎回复快一些? 32-33 token/s

            # from https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device#using-llama.cpp-recommended

                --min-p 0.05 to counteract very rare token predictions - I found this to work well especially for the 1.58bit model.
                --temp 0.6

        # DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf, 33GB model
            # 10 GB per GPU, 21 token/s
            # 回答效果和Q4差不多

        # 加上 --flash-attn可以快1个token/s以内

    3. 使用llama.cpp自带的benchmarking tool测速度

        # 这个评测同样的机器不同时间跑差别挺大的
        # https://github.com/ggerganov/llama.cpp/blob/master/examples/llama-bench/README.md

        (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ llama.cpp/llama-bench --flash-attn 1 --model models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf

            | model                          |       size |     params | backend    | ngl |          test |                  t/s |
            | ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
            | qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |         pp512 |        988.46 ± 2.69 |
            | qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |         tg128 |         34.19 ± 0.16 |

        # pp512: prompt processing with 512 token; tg128: token generation to 128 tokens
        # flash-attn 1有34.74 token/s, 没有的话是34.19 token/s，和开服务器测试差不多

```

+ 用ollama
```
    1. 下载安装
        $ curl -fsSL https://ollama.com/install.sh | sh

    2. 建立模型
        DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.Modelfile:

            FROM /mnt/ssd3/junweil/deepseek/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf
            PARAMETER num_gpu 65
            PARAMETER num_ctx 4096
            TEMPLATE "<｜User｜>{{ .Prompt }}<｜Assistant｜>"

        (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek/models$ ollama create DeepSeek-R1-Distill-Qwen-32B-Q4_K_M -f DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.Modelfile

    3. 开server

        (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek/models$ ollama run DeepSeek-R1-Distill-Qwen-32B-Q4_K_M

        $ OLLAMA_MODEL=DeepSeek-R1-Distill-Qwen-32B-Q4_K_M ollama serve

        关闭需要这样

        $ ollama stop DeepSeek-R1-Distill-Qwen-32B-Q4_K_M
        # 遇到 127.0.0.1:11434: bind: address already in use
        $ sudo systemctl stop ollama

        # 安装方便，但是看不到log，看不到token/s，也不知道怎么设置4卡做tensor parallelism，server port只能在11434。算了不用这个了

```

### R1 量化模型
+ 下载
```
    UD-IQ1_M, 文件大小158GB
    # 在一个文件夹里：https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_M
        (base) junweil@home-lab:/mnt/nvme2/junweil/deepseek$ huggingface-cli download unsloth/DeepSeek-R1-GGUF --include "DeepSeek-R1-UD-IQ1_M/*" --local-dir ./DeepSeek-R1-UD-IQ1_M

    Q4_K_M，文件大小404GB
        (base) junweil@home-lab:/mnt/nvme2/junweil/deepseek$ huggingface-cli download bartowski/DeepSeek-R1-GGUF --include "DeepSeek-R1-Q4_K_M/*" --local-dir ./DeepSeek-R1-Q4_K_M
```

+ 跑量化版本
```
    # tutorial (单卡4090): https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device#using-llama.cpp-recommended
    # tutorail (4卡4090): https://snowkylin.github.io/blogs/a-note-on-deepseek-r1.html#fn:1

    # UD-IQ1_M 模型大小 158 GB
        # 先跑个benchmark
            # 第一次load要10分钟

            (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ llama.cpp/llama-bench --flash-attn 0 --model models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf --n-gpu-layers 27
            # --n-gpu-layers 28: OOM

           | model                          |       size |     params | backend    | ngl |          test |                  t/s |
            | ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
            | deepseek2 671B IQ1_S - 1.5625 bpw | 157.31 GiB |   671.03 B | CUDA       |  27 |         pp512 |         43.38 ± 0.07 |
            | deepseek2 671B IQ1_S - 1.5625 bpw | 157.31 GiB |   671.03 B | CUDA       |  27 |         tg128 |          3.76 ± 0.00 |

            # --flash-attn no improvement
            # --prio 3 no improvement
            # --mmap 0: 显存使用19GB x4, RAM使用90 GB; prompt processing 更快了

                | model                          |       size |     params | backend    | ngl | mmap |          test |                  t/s |
                | ------------------------------ | ---------: | ---------: | ---------- | --: | ---: | ------------: | -------------------: |
                | deepseek2 671B IQ1_S - 1.5625 bpw | 157.31 GiB |   671.03 B | CUDA       |  27 |    0 |         pp512 |         74.39 ± 0.29 |
                | deepseek2 671B IQ1_S - 1.5625 bpw | 157.31 GiB |   671.03 B | CUDA       |  27 |    0 |         tg128 |          3.15 ± 0.04 |

            # --no-kv-offload 1: 慢很多！

        # 跑server

            (deepseek) junweil@ai-precog-machine5:/mnt/ssd3/junweil/deepseek$ ./llama.cpp/llama-server --model models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf --port 8888 --ctx-size 2048 --n-gpu-layers 27

            # 150 秒左右才出非思考token, 3.2 token/s
            # CPU 50%利用率，4xgpu利用率3%, RAM几乎没用

            # --no-mmap, RAM用了98GB
                # 2.5 token/s，更慢了

```

### 构建SGLang + LLama.cpp docker image
```
    # 从我们之前的docker开始, on 4x3090 machine
        $ sudo docker pull junweiliang/base:v1.4
        $ sudo docker run -it --shm-size=128g --gpus all --network=host junweiliang/base:v1.4 bash

        # docker 内的nvidia-smi用的是物理机器上的

        # 原本image的python==3.8，我们安装一下Anaconda, llama.cpp需要cuda

        # so we need
            (sglang) root@ai-precog-machine5:/workspace/Downloads# ls
                Anaconda3-2024.02-1-Linux-x86_64.sh  cuda_12.8.0_570.86.10_linux.run

        # 1. 安装SGLang
            # install conda
                (sglang) root@ai-precog-machine5:/workspace/Downloads# bash Anaconda3-2024.02-1-Linux-x86_64.sh
                ...
                $ source ~/.bashrc
            # install SGLang
                conda create -n sglang python=3.10
                ...
                # follow the pip install from here: https://docs.sglang.ai/start/install.html

        # 2. 安装llama.cpp
            # install cuda 12.8
                (sglang) root@ai-precog-machine5:/workspace/Downloads# bash cuda_12.8.0_570.86.10_linux.run
            # add the nvcc path to $PATH in source

            $ apt install curl libcurl4-openssl-dev
            (sglang) root@ai-precog-machine5:/workspace# git clone https://github.com/ggerganov/llama.cpp
            (sglang) root@ai-precog-machine5:/workspace# cmake llama.cpp -B llama.cpp/build     -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
            (sglang) root@ai-precog-machine5:/workspace# cmake --build llama.cpp/build --config Release -j 8 --clean-first
            (sglang) root@ai-precog-machine5:/workspace# cp llama.cpp/build/bin/llama-* llama.cpp

        # 3. 安装Open WebUI

            (base) root@ai-precog-machine5:/workspace# conda create -n openweb python=3.11
            (openweb) root@ai-precog-machine5:/workspace# python3 -m pip install open-webui

            # run it the first time to download some sample models

        # commit and push the docker image to docker hub

            (base) junweil@ai-precog-machine5:~$ sudo docker commit a680398a17cd junweiliang/base:v2.0
            $ sudo docker login -u "junweiliang" -p "***" docker.io
            $ sudo docker image push junweiliang/base:v2.0

            # you can see this on dockerHub: https://hub.docker.com/r/junweiliang/base/tags
                # 40 GB !!

                # save the docker image into a file, and load it into the cluster machine

                    $ sudo docker save junweiliang/base:v2.0 > docker.base.v2.0.tar


        # so this docker image, we have two env, sglang / openweb
            # sglang env has python3.10 and pytorch
                # pytorch=='2.5.1+cu124'
                # sglang=='0.4.2.post2'
            # openweb env has python3.11
                # pytorch=='2.6.0+cu124'

        # Run it on any other machine
            $ sudo docker pull junweiliang/base:v2.0

            # start docker container (we can run the backend and frontend in one container. Otherwise you need to run two container in one docker network)

                $ sudo docker run -it --rm --shm-size=128g --gpus all --ipc=host -p 8080:8080 -v /mnt/ssd1/junweil/deepseek/:/deepseek junweiliang/base:v2.0 bash

                # see here, we only need 8080 for the web server for outside to call

            # now we are going to start a screen inside the docker container
                # if you have already started one screen before running the docker, now you have nested screen.
                    # to exit back to the docker container screen, you need to press ctr+a then a then d

                # 1. start backend in a screen
                    (base) root@4f6ed8d5262b:/workspace# screen -R deepseek


                    # with DeepSeek-R1-Distill-Qwen-32B
                        (sglang) root@cfadc8fa8aaf:/deepseek# python -m sglang.launch_server --model-path models/DeepSeek-R1-Distill-Qwen-32B --tp 4 --enable-p2p-check --host 0.0.0.0 --port 6666 --trust-remote-code --mem-fraction-static 0.8 --enable-torch-compile

                            # note that we need --mem-fraction-static 0.8 to account for small docker GPU overhead

                    # or with llama.cpp with GGUF quantized model
                        (sglang) root@4f6ed8d5262b:/deepseek# /workspace/llama.cpp/llama-server --model models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --port 6666 --ctx-size 8192 --n-gpu-layers 65

                    press ctr+a then a then d to exit to docker container

                # 2. start the frontend web server

                    (base) root@4f6ed8d5262b:/workspace# screen -R openweb

                    (openweb) root@4f6ed8d5262b:/workspace# export OPENAI_API_BASE_URL=http://127.0.0.1:6666/v1
                    (openweb) root@4f6ed8d5262b:/workspace# export ENABLE_OLLAMA_API=False
                    (openweb) root@4f6ed8d5262b:/workspace# open-webui serve

                    press ctr+a then a then d to exit to docker container

                # you have two screen session running inside the docker

                    (base) root@4f6ed8d5262b:/workspace# screen -list
                    There are screens on:
                            3655.openweb    (02/07/25 17:14:58)     (Detached)
                            683.deepseek    (02/07/25 17:10:02)     (Detached)

            # test on 4xL40 machine

                (base) junweil@ai-precog-machine10:/mnt/ssd2/junweil/deepseek$ sudo docker run -it --rm --shm-size=512g --gpus all --ipc=host -p 8080:8080 -v /mnt/ssd2/junweil/deepseek/:/deepseek junweiliang/base:v2.0 bash
                    # export NCCL_P2P_DISABLE=1

                    --tp 4: 41 token/s

                    --quantization fp8: 55 token/s
                    --torchao-config fp8wo: 10 token/s - yikes!



        # 在集群跑
            # 过程类似，建立实例时，添加端口映射8080，协议TCP即可，然后看实际8080映射到了哪里，可以浏览器访问

            1. SGLang start server

                记得 export NCCL_P2P_DISABLE=1 不然会卡住

                # 4卡
                (sglang) root@befe0a88bcb5:/remote-home/junweil/deepseek# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path models/DeepSeek-R1-Distill-Qwen-32B --tp 4 --enable-p2p-check --host 0.0.0.0 --port 6666 --trust-remote-code --enable-torch-compile

                    # torch compile takes 10 minutes
                    # 36 token/s

                # 8卡
                --tp 8
                    # 55 token/s

            2. llama.cpp

                (sglang) root@befe0a88bcb5:/remote-home/junweil/deepseek# /workspace/llama.cpp/llama-server --model models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --port 6666 --ctx-size 8192 --n-gpu-layers 99 --flash-attn
                    # 27 token/s

                --split-mode none: 放一张卡上，21 GB显存占用, 29 token/s
                # 4 卡： CUDA_VISIBLE_DEVICES=0,1,2,3
                    # 28 token/s

                # 跑 DeepSeek-R1-UD-IQ1_M
                    (sglang) root@befe0a88bcb5:/remote-home/junweil/deepseek# /workspace/llama.cpp/llama-server --model models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf --port 6666 --ctx-size 8192 --n-gpu-layers 99 --flash-attn

                    # load for 4 minutes
                    # 26GBx8, 18 token/s
                    # 4卡 OOM

                    # benchmark

                        (sglang) root@befe0a88bcb5:/remote-home/junweil/deepseek# /workspace/llama.cpp/llama-bench --flash-attn 1 --model models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf

                        # 20 token/s

                        (sglang) root@befe0a88bcb5:/remote-home/junweil/deepseek# /workspace/llama.cpp/llama-bench --flash-attn 1 --model models/DeepSeek-R1-IQ4_XS/DeepSeek-R1-IQ4_XS/DeepSeek-R1-IQ4_XS-00001-of-00010.gguf

                        # 16.7 token/s
                            # -nkvo 1: 10 token/s # this saves some memory but very slow

                        DeepSeek-R1-Q4_K_M failed to load model

                # 跑 DeepSeek-R1-Q4_K_M OOM

                # 跑 DeepSeek-R1-IQ4_XS
                    (sglang) root@befe0a88bcb5:/remote-home/junweil/deepseek# /workspace/llama.cpp/llama-server --model models/DeepSeek-R1-IQ4_XS/DeepSeek-R1-IQ4_XS/DeepSeek-R1-IQ4_XS-00001-of-00010.gguf --port 6666 --ctx-size 1024 --n-gpu-layers 99 --flash-attn
                        # 2048以上都OOM, 用1024可以

                        # 15 token/s
```

+ Jetson系列
```
    # Orin AGX 64 GB, JetPack 5
        # Ubuntu 20.04.6 LTS (GNU/Linux 5.10.120-tegra aarch64)

        # JetPack 6.x?
            # instruction: https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html

            # Need one Ubuntu host machine, download nvidia SDK manage
                # https://developer.nvidia.com/sdk-manager#installation_get_started


        # 安装package

            # 1. Anaconda for Jetson
                $ wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-aarch64.sh

            # 2. SGLang
                # 似乎需要JetPack 6.x，算了
                # https://github.com/shahizat/SGLang-Jetson

                $ sudo apt-get install libopenblas-dev



            # 3. llama.cpp
                # JetPack 5 has CUDA 11.4

                # 1. 安装 cmake curl

                    (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ wget https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-linux-aarch64.sh

                    $ sudo bash cmake-3.31.5-linux-aarch64.sh

                    (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ sudo ln -s /mnt/nvme1/junweil/deepseek/cmake-3.31.5-linux-aarch64/bin/* /usr/local/bin

                    (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ sudo apt update && sudo apt upgrade && sudo apt install curl libssl-dev libcurl4-openssl-dev


                # 2. build!

                    (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ cmake llama.cpp -B llama.cpp/build     -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=native

                    (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ cmake --build llama.cpp/build --config Release -j 8 --clean-first

                    (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ cp llama.cpp/build/bin/llama-* llama.cpp


            # 4. Open WebUI

                $ conda create -n openweb python=3.11

                # somehow we need this
                $ curl https://sh.rustup.rs -sSf | sh
                # reopen terminal
                $sudo apt-get install libclang-dev

                $ python3 -m pip install open-webui

        # 运行测试 - 32B

            # 量化模型： /mnt/nvme1/junweil/deepseek/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf

                1. benchmark:

                    MODE_30w
                    $ sudo nvpmodel -m 2
                    $ sudo nvpmodel -m 3 50W
                    30W to 50W, token/s 5.37 -> 5.39, no difference

                    (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ llama.cpp/llama-bench --flash-attn 1 --model DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf

                        ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
                        ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
                        ggml_cuda_init: found 1 CUDA devices:
                          Device 0: Orin, compute capability 8.7, VMM: yes
                        | model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
                        | ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
                        | qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |  1 |         pp512 |        128.64 ± 0.03 |
                        | qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |  1 |         tg128 |          5.37 ± 0.01 |

                        build: 55ac8c77 (4675)

                        --flash-attn 0: 5.26

                    $ sudo nvpmodel -m 0 MAX power mode, (60W?)

                        | model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
                        | ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
                        | qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |  1 |         pp512 |        201.73 ± 0.11 |
                        | qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |  1 |         tg128 |          6.84 ± 0.01 |

                        # DeepSeek-R1-Distill-Qwen-32B-IQ4_NL.gguf
                            not faster

                        # DeepSeek-R1-Distill-Qwen-32B-IQ3_XS.gguf
                            not faster
                        # DeepSeek-R1-Distill-Qwen-32B-Q2_K.gguf
                            not faster!!!

                2. Run!
                    # start backend

                        (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ ./llama.cpp/llama-server --model DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --port 8888 --ctx-size 8192 --n-gpu-layers 65 --host 0.0.0.0

                        # need --host 0.0.0.0 to be accessed from another machine

                    # start frontend
                        # AGX got this error
                            ImportError: /home/junweil/anaconda3/envs/openweb/lib/python3.11/site-packages/chroma_hnswlib.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block

                        # this solves it
                            (openweb) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ export LD_PRELOAD=/home/junweil/anaconda3/envs/openweb/lib/python3.11/site-packages/chroma_hnswlib.libs/libgomp-d22c30c5.so.1.0.0

                        $ open-webui serve


                     # start frontend on another machine

                        (openweb) junweil@home-lab:~$ export ENABLE_OLLAMA_API=False
                        (openweb) junweil@home-lab:~$ export OPENAI_API_BASE_URL=http://10.13.11.212:8888/v1
                        (openweb) junweil@home-lab:~$ open-webui serve

        # 运行测试 - 14B模型:
            1. benchmark
                (base) junweil@ai-precog-agx1:/mnt/nvme1/junweil/deepseek$ llama.cpp/llama-bench --flash-attn 1 --model DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf

                | model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
                | ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
                | qwen2 14B Q4_K - Medium        |   8.37 GiB |    14.77 B | CUDA       |  99 |  1 |         pp512 |        455.47 ± 0.10 |
                | qwen2 14B Q4_K - Medium        |   8.37 GiB |    14.77 B | CUDA       |  99 |  1 |         tg128 |         14.49 ± 0.01 |

            "How many Rs are there in strawberry?" 这个问题回答错误！让它重新数一次才数对

            DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf 一次就数对了。12 token/s

```

