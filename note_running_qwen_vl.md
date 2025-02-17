# note on running Qwen VL models

+ Using SGLang
```
    # download model
        $ huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct --local-dir ./Qwen2.5-VL-72B-Instruct

    # need sglang>0.4.3
    # 升级
        $ pip install sgl-kernel --force-reinstall --no-deps
        $ pip install transformers==4.48.3

        # need to install the main branch
            (deepseek) junweil@ai-precog-machine4:/mnt/ssd1/junweil/qwen$ git clone -b main https://github.com/sgl-project/sglang.git

            (deepseek) junweil@ai-precog-machine4:/mnt/ssd1/junweil/qwen/sglang$ pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

    # backend server!

        # Qwen2.5-VL-7B-Instruct

        (deepseek) junweil@ai-precog-machine4:/mnt/ssd1/junweil/qwen/models$ python -m sglang.launch_server --model-path Qwen2.5-VL-7B-Instruct/ --tp 4 --enable-p2p-check --host 0.0.0.0 --port 7777 --trust-remote-code --chat-template qwen2-vl

        # 112 token/s

        # takes 23 GB x4 GPU memory

            --enable-torch-compile # 130 token/s

            4k image will OOM; 1280x960 image is ok for one turn, will OOM second turn

            # 加 --max-prefill 8192 预防OOM?

        # 电梯理解能力不太行，比MGM-34B 差

        # Qwen2.5-VL-72B-Instruct
            # 官方网页版可以直接用来测试，比7B好很多
                # https://chat.qwenlm.ai/

            # 4卡3090 OOM


        # TODO, use GGUF quantized models
            # https://huggingface.co/bartowski/Qwen2-VL-72B-Instruct-GGUF
            # need to wait for llama.cpp to support Qwen2.5 VL, and bartowski to post it
                # https://github.com/ggml-org/llama.cpp/issues/11483

    # front end
        $ export ENABLE_OLLAMA_API=False
        $ export OPENAI_API_BASE_URL=http://127.0.0.1:7777/v1
        $ open-webui serve --port 8081

        # you can then upload (multiple) image and ask questions
```
