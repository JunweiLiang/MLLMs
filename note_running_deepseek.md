# Note for running Deepseek R1 and R1-Qwen-32B


### 有用的链接
+ R1 全量模型下载链接： `https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d`
+ R1 量化版本 - unsloth版，推荐3bit以内用这个: `https://huggingface.co/unsloth/DeepSeek-R1-GGUF`
+ R1 量化版本 - bartowski版，各个bit的都有: `https://huggingface.co/bartowski/DeepSeek-R1-GGUF`
+ R1-Qwen-32B全量版本：`https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/tree/main`
+ R1-Qwen-32B量化版本，目前bartowski的下载比unsloth多：`https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF`
+ 4x4090部署R1 量化版本教程：`https://snowkylin.github.io/blogs/a-note-on-deepseek-r1.html`
+ unsloth的量化版本教程：`https://unsloth.ai/blog/deepseekr1-dynamic`


### 部署教程

测试环境包括：
+ 8卡A6000，共384GB显存VRAM
+ 4卡3090=96GB,  2卡3090=48GB，单卡3090=24GB，笔记本4090=16GB
+ 4卡L40=192GB
+ ARM系列，Orin NX 16 GB, Orin AGX 64 GB
