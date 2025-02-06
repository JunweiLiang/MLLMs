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
+ 4卡3090=96GB,  2卡3090=48GB，单卡3090=24GB，笔记本4090=16GB
+ 4卡L40=192GB
+ ARM系列，Orin NX 16 GB, Orin AGX 64 GB

#### 速度测试总结

| 短序列输入测试                                  |       |                  |           |          |               |                      |                      |
| ---------------------------------------- | ----- | ---------------- | --------- | -------- | ------------- | -------------------- | -------------------- |
| 模型                                       | 模型大小  | 测试机器             | 工具        | 是否开FP8推理 | 使用显存          | 回答延迟（第一个非思考的token时间） | 试过的最好的速度throughput   |
| DeepSeek-R1-Distill-Qwen-32B             | 62GB  | 4x3090, 64核256GB | SGLang    | 否        | 22GB x4       | ~14秒                 | 40 token/s           |
| DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf   | 33GB  | 4x3090, 64核256GB | llama.cpp | 否        | 10GB x4       | ~20秒                 | 21 token/s           |
| DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf | 19GB  | 4x3090, 64核256GB | llama.cpp | 否        | 6GB x4        | ~20秒                 | 32 token/s           |
|                                          |       | 4x3090, 64核256GB | llama.cpp | 否        | 21GBx1        | ~20秒                 | 26 token/s           |
|                                          |       | 4x3090, 64核256GB | SGLang    | 否        | 22GB x4 这个很奇怪 | 0秒                   | 67 token/s，但是有可能胡言乱语 |
| DeepSeek-R1-UD-IQ1_M                     | 158GB | 4x3090, 64核256GB | llama.cpp | 否        | 21GBx4        | ~180秒                | 3.2 token/s          |

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
                $ con
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


        # 用SGLang docker

            $ sudo docker pull lmsysorg/sglang:latest

            (base) junweil@ai-precog-machine5:~$ sudo docker run -it --shm-size=128g --gpus all --ipc=host -p 6666:6666 -v /mnt/ssd3/junweil/deepseek/:/deepseek lmsysorg/sglang:latest bash

                # 报错： nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.5, please update your driver to a newer version,
                # 意味着，这个docker image的cuda比物理机的cuda高，所以要把物理机器的driver和cuda升级一下

                # 更新 nvidia-driver 550 -> 570, CUDA -> 12.8
                $ sudo add-apt-repository ppa:graphics-drivers/ppa
                $ sudo apt-get --purge remove nvidia-*
                $ sudo apt-get --purge remove libnvidia-*
                $ sudo ubuntu-drivers autoinstall

                # then you might need to reinstall and restart docker

                    $ sudo apt-get install nvidia-container-toolkit
                    $ sudo systemctl daemon-reload
                    $ sudo systemctl restart docker

            # run again

                # start docker
                    (base) junweil@ai-precog-machine5:~$ sudo docker run -it --shm-size=128g --gpus all --ipc=host -p 6666:6666 -v /mnt/ssd3/junweil/deepseek/:/deepseek lmsysorg/sglang:latest bash

                # run it in the docker container
                    root@0a5a6ec82561:/deepseek/models/# python3 -m sglang.launch_server --model-path DeepSeek-R1-Distill-Qwen-32B-Q4_K_M --tp 4 --enable-p2p-check --host 0.0.0.0 --port 6666 --trust-remote-code


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


