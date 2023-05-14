# Awesome-llm-and-aigc
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

🔥🔥🔥 This repository lists some awesome public projects about Large Language Model, Vision Foundation Model and AI Generated Content.

## Contents
- [Awesome-llm-and-aigc](#awesome-llm-and-aigc)
  - [Summary](#summary)
    - [Frameworks](#frameworks)
      - [Official Version](#official-version)
        - [Large Language Model](#large-language-model)
        - [Vision Foundation Model](#vision-foundation-model)
        - [AI Generated Content](#ai-generated-content)
      - [C Implementation](#c-implementation)
      - [Rust Implementation](#rust-implementation)
      - [zig Implementation](#zig-implementation)
    - [Awesome List](#awesome-list)
    - [Paper and Code Overview](#paper-and-code-overview)
      - [Paper Review](#paper-review)
      - [Code Review](#code-review)
    - [Learning Resources](#learning-resources)
  - [Prompts](#prompts)
  - [Open API](#open-api)
    - [Python API](#python-api)
    - [Rust API](#rust-api)
    - [Csharp API](#csharp-api)
    - [Node.js API](#node.js-api)
  - [Device Deployment](#device-deployment)
  - [Applications](#applications)
    - [IDE](#ide)
    - [Academic](#academic)
    - [Computer Vision](#computer-vision)
    - [Wechat](#wechat)
    - [Translator](#translator)
    - [GUI](#gui)
  - [Blogs](#blogs)

## Summary

  - ### Frameworks

    - #### Official Version

      - ##### Large Language Model

        - GPT-1 : "Improving Language Understanding by Generative Pre-Training". (**[cs.ubc.ca, 2018](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)**).

        - [GPT-2](https://github.com/openai/gpt-2) <img src="https://img.shields.io/github/stars/openai/gpt-2?style=social"/> : "Language Models are Unsupervised Multitask Learners". (**[OpenAI blog, 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)**). [Better language models and their implications](https://openai.com/research/better-language-models).

        - [GPT-3](https://github.com/openai/gpt-3) <img src="https://img.shields.io/github/stars/openai/gpt-3?style=social"/> : "GPT-3: Language Models are Few-Shot Learners". (**[arXiv 2020](https://arxiv.org/abs/2005.14165)**).

        - InstructGPT : "Training language models to follow instructions with human feedback". (**[arXiv 2022](https://arxiv.org/abs/2203.02155)**). "Aligning language models to follow instructions". (**[OpenAI blog, 2022](https://openai.com/research/instruction-following)**).

        - [ChatGPT](https://chat.openai.com/): [Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt).

        - [GPT-4](https://openai.com/product/gpt-4): GPT-4 is OpenAI’s most advanced system, producing safer and more useful responses.   

        - [Whisper](https://github.com/openai/whisper) <img src="https://img.shields.io/github/stars/openai/whisper?style=social"/> : Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification. "Robust Speech Recognition via Large-Scale Weak Supervision". (**[arXiv 2022](https://arxiv.org/abs/2212.04356)**). 

        - [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) <img src="https://img.shields.io/github/stars/Vision-CAIR/MiniGPT-4?style=social"/> : MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models. [minigpt-4.github.io](https://minigpt-4.github.io/)

        - [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) <img src="https://img.shields.io/github/stars/Significant-Gravitas/Auto-GPT?style=social"/> : Auto-GPT: An Autonomous GPT-4 Experiment. Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. This program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, Auto-GPT pushes the boundaries of what is possible with AI.

        - [LLaMA](https://github.com/facebookresearch/llama) <img src="https://img.shields.io/github/stars/facebookresearch/llama?style=social"/> : Inference code for LLaMA models. "LLaMA: Open and Efficient Foundation Language Models". (**[arXiv 2023](https://arxiv.org/abs/2302.13971)**). 

        - [StableLM](https://github.com/Stability-AI/StableLM) <img src="https://img.shields.io/github/stars/Stability-AI/StableLM?style=social"/> : StableLM: Stability AI Language Models.

        - [JARVIS](https://github.com/microsoft/JARVIS) <img src="https://img.shields.io/github/stars/microsoft/JARVIS?style=social"/> : JARVIS, a system to connect LLMs with ML community. "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace". (**[arXiv 2023](https://arxiv.org/abs/2303.17580)**). 

        - [minGPT](https://github.com/karpathy/minGPT) <img src="https://img.shields.io/github/stars/karpathy/minGPT?style=social"/> : A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training.

        - [nanoGPT](https://github.com/karpathy/nanoGPT) <img src="https://img.shields.io/github/stars/karpathy/nanoGPT?style=social"/> : The simplest, fastest repository for training/finetuning medium-sized GPTs. 

        - [Claude](https://www.anthropic.com/product) : Claude is a next-generation AI assistant based on Anthropic’s research into training helpful, honest, and harmless AI systems. 

        - [MicroGPT](https://github.com/muellerberndt/micro-gpt) <img src="https://img.shields.io/github/stars/muellerberndt/micro-gpt?style=social"/> : A simple and effective autonomous agent compatible with GPT-3.5-Turbo and GPT-4. MicroGPT aims to be as compact and reliable as possible.

        - [Dolly](https://github.com/databrickslabs/dolly) <img src="https://img.shields.io/github/stars/databrickslabs/dolly?style=social"/> : Databricks’ Dolly, a large language model trained on the Databricks Machine Learning Platform. [Hello Dolly: Democratizing the magic of ChatGPT with open models](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)

        - [LMFlow](https://github.com/OptimalScale/LMFlow) <img src="https://img.shields.io/github/stars/OptimalScale/LMFlow?style=social"/> : An extensible, convenient, and efficient toolbox for finetuning large machine learning models, designed to be user-friendly, speedy and reliable, and accessible to the entire community. Large Language Model for All. [optimalscale.github.io/LMFlow/](https://optimalscale.github.io/LMFlow/)

        - [Open-Assistant](https://github.com/LAION-AI/Open-Assistant) <img src="https://img.shields.io/github/stars/LAION-AI/Open-Assistant?style=social"/> : OpenAssistant is a chat-based assistant that understands tasks, can interact with third-party systems, and retrieve information dynamically to do so. [open-assistant.io](https://open-assistant.io/)

        - [Colossal-AI](https://github.com/hpcaitech/ColossalAI) <img src="https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social"/> : Making big AI models cheaper, easier, and scalable. [www.colossalai.org](www.colossalai.org). "Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training". (**[arXiv 2021](https://arxiv.org/abs/2110.14883)**). 

        - [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama) <img src="https://img.shields.io/github/stars/Lightning-AI/lit-llama?style=social"/> : ⚡ Lit-LLaMA. Implementation of the LLaMA language model based on nanoGPT. Supports flash attention, Int8 and GPTQ 4bit quantization, LoRA and LLaMA-Adapter fine-tuning, pre-training. Apache 2.0-licensed. 

        - [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) <img src="https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM?style=social"/> : "Instruction Tuning with GPT-4". (**[arXiv 2023](https://arxiv.org/abs/2304.03277)**). [instruction-tuning-with-gpt-4.github.io/](https://instruction-tuning-with-gpt-4.github.io/) 

        - [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) <img src="https://img.shields.io/github/stars/THUDM/ChatGLM-6B?style=social"/> : ChatGLM-6B：开源双语对话语言模型 | An Open Bilingual Dialogue Language Model. ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。 "GLM: General Language Model Pretraining with Autoregressive Blank Infilling". (**[ACL 2022](https://aclanthology.org/2022.acl-long.26/)**).  "GLM-130B: An Open Bilingual Pre-trained Model". (**[ICLR 2023](https://openreview.net/forum?id=-Aw0rrrPUF)**). 

        - [MOSS](https://github.com/OpenLMLab/MOSS) <img src="https://img.shields.io/github/stars/OpenLMLab/MOSS?style=social"/> : An open-source tool-augmented conversational language model from Fudan University. MOSS是一个支持中英双语和多种插件的开源对话语言模型，moss-moon系列模型具有160亿参数，在FP16精度下可在单张A100/A800或两张3090显卡运行，在INT4/8精度下可在单张3090显卡运行。MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。[txsun1997.github.io/blogs/moss.html](https://txsun1997.github.io/blogs/moss.html)

        - [百度-文心一言](https://yiyan.baidu.com/welcome) : 百度全新一代知识增强大语言模型，文心大模型家族的新成员，能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。

        - [阿里云-通义千问](https://tongyi.aliyun.com/) : 通义千问，是阿里云推出的一个超大规模的语言模型，功能包括多轮对话、文案创作、逻辑推理、多模态理解、多语言支持。能够跟人类进行多轮的交互，也融入了多模态的知识理解，且有文案创作能力，能够续写小说，编写邮件等。

        - [商汤科技-日日新SenseNova](https://techday.sensetime.com/?utm_source=baidu-sem-pc&utm_medium=cpc&utm_campaign=PC-%E6%8A%80%E6%9C%AF%E4%BA%A4%E6%B5%81%E6%97%A5-%E4%BA%A7%E5%93%81%E8%AF%8D-%E6%97%A5%E6%97%A5%E6%96%B0&utm_content=%E6%97%A5%E6%97%A5%E6%96%B0&utm_term=%E6%97%A5%E6%97%A5%E6%96%B0SenseNova&e_creative=73937788324&e_keywordid=594802524403) : 日日新（SenseNova），是商汤科技宣布推出的大模型体系，包括自然语言处理模型“商量”（SenseChat）、文生图模型“秒画”和数字人视频生成平台“如影”（SenseAvatar）等。

        - [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese) <img src="https://img.shields.io/github/stars/Morizeyao/GPT2-Chinese?style=social"/> : Chinese version of GPT2 training code, using BERT tokenizer. 
        
        - [feizc/Visual-LLaMA](https://github.com/feizc/Visual-LLaMA) <img src="https://img.shields.io/github/stars/feizc/Visual-LLaMA?style=social"/> : Open LLaMA Eyes to See the World. This project aims to optimize LLaMA model for visual information understanding like GPT-4 and further explore the potentional of large language model.

        - [Lightning-AI/lightning-colossalai](https://github.com/Lightning-AI/lightning-colossalai) <img src="https://img.shields.io/github/stars/Lightning-AI/lightning-colossalai?style=social"/> : Efficient Large-Scale Distributed Training with [Colossal-AI](https://colossalai.org/) and [Lightning AI](https://lightning.ai/).




      - ##### Vision Foundation Model

        - [InternImage](https://github.com/OpenGVLab/InternImage) <img src="https://img.shields.io/github/stars/OpenGVLab/InternImage?style=social"/> : "InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions". (**[CVPR 2023](https://arxiv.org/abs/2211.05778)**). 

        - [DINOv2](https://github.com/facebookresearch/dinov2) <img src="https://img.shields.io/github/stars/facebookresearch/dinov2?style=social"/> : "DINOv2: Learning Robust Visual Features without Supervision". (**[arXiv 2023](https://arxiv.org/abs/2304.07193)**). 

        - [Segment Anything](https://github.com/facebookresearch/segment-anything) <img src="https://img.shields.io/github/stars/facebookresearch/segment-anything?style=social"/> : The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. "Segment Anything". (**[arXiv 2023](https://arxiv.org/abs/2304.02643)**). 

        - [Track-Anything](https://github.com/gaomingqi/Track-Anything) <img src="https://img.shields.io/github/stars/gaomingqi/Track-Anything?style=social"/> : Track-Anything is an Efficient Development Toolkit for Video Object Tracking and Segmentation, based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem).

        - [LLaVA](https://github.com/haotian-liu/LLaVA) <img src="https://img.shields.io/github/stars/haotian-liu/LLaVA?style=social"/> : 🌋 LLaVA: Large Language and Vision Assistant. Visual instruction tuning towards large language and vision models with GPT-4 level capabilities. [llava.hliu.cc](https://llava.hliu.cc/). "Visual Instruction Tuning". (**[arXiv 2023](https://arxiv.org/abs/2304.08485)**). 







      - ##### AI Generated Content

        - [Stable Diffusion](https://github.com/CompVis/stable-diffusion) <img src="https://img.shields.io/github/stars/CompVis/stable-diffusion?style=social"/> : Stable Diffusion is a latent text-to-image diffusion model. Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work "High-Resolution Image Synthesis with Latent Diffusion Models". (**[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)**). 

        - [Stable Diffusion Version 2](https://github.com/Stability-AI/stablediffusion) <img src="https://img.shields.io/github/stars/Stability-AI/stablediffusion?style=social"/> : This repository contains [Stable Diffusion](https://github.com/CompVis/stable-diffusion) models trained from scratch and will be continuously updated with new checkpoints. "High-Resolution Image Synthesis with Latent Diffusion Models". (**[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)**). 

        - [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) <img src="https://img.shields.io/github/stars/microsoft/visual-chatgpt?style=social"/> : Visual ChatGPT connects ChatGPT and a series of Visual Foundation Models to enable sending and receiving images during chatting. "Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models". (**[arXiv 2023](https://arxiv.org/abs/2303.04671)**). 

        - [Firefly](https://www.adobe.com/sensei/generative-ai/firefly.html) : Adobe Firefly: Experiment, imagine, and make an infinite range of creations with Firefly, a family of creative generative AI models coming to Adobe products.

        - [AudioGPT](https://github.com/AIGC-Audio/AudioGPT) <img src="https://img.shields.io/github/stars/AIGC-Audio/AudioGPT?style=social"/> : AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head.

        - [PandasAI](https://github.com/gventuri/pandas-ai) <img src="https://img.shields.io/github/stars/gventuri/pandas-ai?style=social"/> : Pandas AI is a Python library that adds generative artificial intelligence capabilities to Pandas, the popular data analysis and manipulation tool. It is designed to be used in conjunction with Pandas, and is not a replacement for it.



    - #### C Implementation

      - [llama.cpp](https://github.com/ggerganov/llama.cpp) <img src="https://img.shields.io/github/stars/ggerganov/llama.cpp?style=social"/> : Inference of [LLaMA](https://github.com/facebookresearch/llama) model in pure C/C++.

      - [skeskinen/llama-lite](https://github.com/skeskinen/llama-lite) <img src="https://img.shields.io/github/stars/skeskinen/llama-lite?style=social"/> : Embeddings focused small version of Llama NLP model.

      - [whisper.cpp](https://github.com/ggerganov/whisper.cpp) <img src="https://img.shields.io/github/stars/ggerganov/whisper.cpp?style=social"/> : High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model.

      - [Const-me/Whisper](https://github.com/Const-me/Whisper) <img src="https://img.shields.io/github/stars/Const-me/Whisper?style=social"/> : High-performance GPGPU inference of OpenAI's Whisper automatic speech recognition (ASR) model.






    - #### Rust Implementation

      - [rustformers/llm](https://github.com/rustformers/llm) <img src="https://img.shields.io/github/stars/rustformers/llm?style=social"/> : Run inference for Large Language Models on CPU, with Rust 🦀🚀🦙.

      - [Noeda/rllama](https://github.com/Noeda/rllama) <img src="https://img.shields.io/github/stars/Noeda/rllama?style=social"/> : Rust+OpenCL+AVX2 implementation of LLaMA inference code.

      - [Atome-FE/llama-node](https://github.com/Atome-FE/llama-node) <img src="https://img.shields.io/github/stars/Atome-FE/llama-node?style=social"/> : Believe in AI democratization. llama for nodejs backed by llama-rs and llama.cpp, work locally on your laptop CPU. support llama/alpaca/gpt4all/vicuna model. [www.npmjs.com/package/llama-node](https://www.npmjs.com/package/llama-node)

      - [tazz4843/whisper-rs](https://github.com/tazz4843/whisper-rs) <img src="https://img.shields.io/github/stars/tazz4843/whisper-rs?style=social"/> : Rust bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

   
    - #### Zig Implementation
    
      - [renerocksai/gpt4all.zig](https://github.com/renerocksai/gpt4all.zig) <img src="https://img.shields.io/github/stars/renerocksai/gpt4all.zig?style=social"/> : ZIG build for a terminal-based chat client for an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa.

    

  - ### Awesome List

    - [formulahendry/awesome-gpt](https://github.com/formulahendry/awesome-gpt) <img src="https://img.shields.io/github/stars/formulahendry/awesome-gpt?style=social"/> : A curated list of awesome projects and resources related to GPT, ChatGPT, OpenAI, LLM, and more.  

    - [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) <img src="https://img.shields.io/github/stars/Hannibal046/Awesome-LLM?style=social"/> : Awesome-LLM: a curated list of Large Language Model.

    - [cedrickchee/awesome-transformer-nlp](https://github.com/cedrickchee/awesome-transformer-nlp) <img src="https://img.shields.io/github/stars/cedrickchee/awesome-transformer-nlp?style=social"/> : A curated list of NLP resources focused on Transformer networks, attention mechanism, GPT, BERT, ChatGPT, LLMs, and transfer learning. 

    - [GT-RIPL/Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics) <img src="https://img.shields.io/github/stars/GT-RIPL/Awesome-LLM-Robotics?style=social"/> : A comprehensive list of papers using large language/multi-modal models for Robotics/RL, including papers, codes, and related websites.

    - [mikhail-bot/awesome-gpt3](https://github.com/mikhail-bot/awesome-gpt3) <img src="https://img.shields.io/github/stars/mikhail-bot/awesome-gpt3?style=social"/> :A Curated list of awesome GPT3 tools, libraries and resources.

    - [imaurer/awesome-decentralized-llm](https://github.com/imaurer/awesome-decentralized-llm) <img src="https://img.shields.io/github/stars/imaurer/awesome-decentralized-llm?style=social"/> : Repos and resources for running LLMs locally. (e.g. LLaMA, Cerebras, RWKV).

    - [csbl-br/awesome-compbio-chatgpt](https://github.com/csbl-br/awesome-compbio-chatgpt) <img src="https://img.shields.io/github/stars/csbl-br/awesome-compbio-chatgpt?style=social"/> : An awesome repository of community-curated applications of ChatGPT and other LLMs in computational biology!

    - [atfortes/LLM-Reasoning-Papers](https://github.com/atfortes/LLM-Reasoning-Papers) <img src="https://img.shields.io/github/stars/atfortes/LLM-Reasoning-Papers?style=social"/> : Collection of papers and resources on Reasoning in Large Language Models (LLMs), including Chain-of-Thought (CoT), Instruction-Tuning, and others. 

    - [yzfly/Awesome-AGI](https://github.com/yzfly/Awesome-AGI) <img src="https://img.shields.io/github/stars/yzfly/Awesome-AGI?style=social"/> : A curated list of awesome AGI frameworks, software and resources.

    - [xx025/carrot](https://github.com/xx025/carrot) <img src="https://img.shields.io/github/stars/xx025/carrot?style=social"/> : Free ChatGPT Site List. [cc.ai55.cc](https://cc.ai55.cc/)

    - [LiLittleCat/awesome-free-chatgpt](https://github.com/LiLittleCat/awesome-free-chatgpt) <img src="https://img.shields.io/github/stars/LiLittleCat/awesome-free-chatgpt?style=social"/> : 🆓免费的 ChatGPT 镜像网站列表，持续更新。List of free ChatGPT mirror sites, continuously updated. 

    - [lzwme/chatgpt-sites](https://github.com/lzwme/chatgpt-sites) <img src="https://img.shields.io/github/stars/lzwme/chatgpt-sites?style=social"/> : 搜集国内可用的 ChatGPT 在线体验免费网站列表。定时任务每日更新。[lzw.me/x/chatgpt-sites/](https://lzw.me/x/chatgpt-sites/)

    - [steven2358/awesome-generative-ai](https://github.com/steven2358/awesome-generative-ai) <img src="https://img.shields.io/github/stars/steven2358/awesome-generative-ai?style=social"/> : A curated list of modern Generative Artificial Intelligence projects and services.

    - [wshzd/Awesome-AIGC](https://github.com/wshzd/Awesome-AIGC) <img src="https://img.shields.io/github/stars/wshzd/Awesome-AIGC?style=social"/> : AIGC资料汇总学习，持续更新...... 

    - [doanbactam/awesome-stable-diffusion](https://github.com/doanbactam/awesome-stable-diffusion) <img src="https://img.shields.io/github/stars/doanbactam/awesome-stable-diffusion?style=social"/> : A curated list of awesome stable diffusion resources 🌟 

    - [Yutong-Zhou-cv/Awesome-Text-to-Image](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image) <img src="https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Text-to-Image?style=social"/> : (ෆ`꒳´ෆ) A Survey on Text-to-Image Generation/Synthesis. 





  - ### Paper and Code Overview

    - #### Paper Review

      - [daochenzha/data-centric-AI](https://github.com/daochenzha/data-centric-AI) <img src="https://img.shields.io/github/stars/daochenzha/data-centric-AI?style=social"/> : A curated, but incomplete, list of data-centric AI resources. "Data-centric Artificial Intelligence: A Survey". (**[arXiv 2023](https://arxiv.org/abs/2303.10158)**).

      - [KSESEU/LLMPapers](https://github.com/KSESEU/LLMPapers) <img src="https://img.shields.io/github/stars/KSESEU/LLMPapers?style=social"/> : Collection of papers and related works for Large Language Models (ChatGPT, GPT-3, Codex etc.).


    - #### Code Review

      - [GPT4All](https://github.com/nomic-ai/gpt4all) <img src="https://img.shields.io/github/stars/nomic-ai/gpt4all?style=social"/> : gpt4all: an ecosystem of open-source chatbots trained on a massive collections of clean assistant data including code, stories and dialogue. Demo, data, and code to train open-source assistant-style large language model based on GPT-J and LLaMa.

      - [1595901624/gpt-aggregated-edition](https://github.com/1595901624/gpt-aggregated-edition) <img src="https://img.shields.io/github/stars/1595901624/gpt-aggregated-edition?style=social"/> : 聚合ChatGPT官方版、ChatGPT免费版、文心一言、Poe、chatchat等多平台，支持自定义导入平台。

      - [FreedomIntelligence/LLMZoo](https://github.com/FreedomIntelligence/LLMZoo) <img src="https://img.shields.io/github/stars/FreedomIntelligence/LLMZoo?style=social"/> : ⚡LLM Zoo is a project that provides data, models, and evaluation benchmark for large language models.⚡ [Tech Report](https://github.com/FreedomIntelligence/LLMZoo/blob/main/assets/llmzoo.pdf)

      - [shm007g/LLaMA-Cult-and-More](https://github.com/shm007g/LLaMA-Cult-and-More) <img src="https://img.shields.io/github/stars/shm007g/LLaMA-Cult-and-More?style=social"/> : News about 🦙 Cult and other AIGC models.

      - [sobelio/llm-chain](https://github.com/sobelio/llm-chain) <img src="https://img.shields.io/github/stars/sobelio/llm-chain?style=social"/> : llm-chain is a collection of Rust crates designed to help you work with Large Language Models (LLMs) more effectively. [llm-chain.xyz](https://llm-chain.xyz/)

      - [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) <img src="https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca?style=social"/> : 中文LLaMA&Alpaca大语言模型+本地CPU/GPU部署 (Chinese LLaMA & Alpaca LLMs). 本项目开源了中文LLaMA模型和指令精调的Alpaca大模型。这些模型在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力。"Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca". (**[arXiv 2023](https://arxiv.org/abs/2304.08177)**). 



  - ### Learning Resources
    - [Microsoft OpenAI](https://aka.ms/cn/LearnOpenAI): Microsoft OpenAI 学习工具包。




## Prompts

  - [f/awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts) <img src="https://img.shields.io/github/stars/f/awesome-chatgpt-prompts?style=social"/> : This repo includes ChatGPT prompt curation to use ChatGPT better. 

  - [travistangvh/ChatGPT-Data-Science-Prompts](https://github.com/travistangvh/ChatGPT-Data-Science-Prompts) <img src="https://img.shields.io/github/stars/travistangvh/ChatGPT-Data-Science-Prompts?style=social"/> : 🚀 ChatGPT Prompts for Data Science! A repository of 60 useful data science prompts for ChatGPT.

  - [kevinamiri/Instructgpt-prompts](https://github.com/kevinamiri/Instructgpt-prompts) <img src="https://img.shields.io/github/stars/kevinamiri/Instructgpt-prompts?style=social"/> : A collection of ChatGPT and GPT-3.5 instruction-based prompts for generating and classifying text. [prompts.maila.ai/](https://prompts.maila.ai/)





## Open API




  - ### Python API

    - [gpt4free](https://github.com/xtekky/gpt4free) <img src="https://img.shields.io/github/stars/xtekky/gpt4free?style=social"/> : decentralising the Ai Industry, just some language model api's... [discord.gg/gpt4free](https://discord.gg/gpt4free)

    - [acheong08/ChatGPT](https://github.com/acheong08/ChatGPT) <img src="https://img.shields.io/github/stars/acheong08/ChatGPT?style=social"/> : Reverse Engineered ChatGPT API by OpenAI. Extensible for chatbots etc.

    - [wong2/chatgpt-google-extension](https://github.com/wong2/chatgpt-google-extension) <img src="https://img.shields.io/github/stars/wong2/chatgpt-google-extension?style=social"/> : A browser extension that enhance search engines with ChatGPT.

    - [acheong08/EdgeGPT](https://github.com/acheong08/EdgeGPT) <img src="httccccccps://img.shields.io/github/stars/acheong08/EdgeGPT?style=social"/> : Reverse engineered API of Microsoft's Bing Chat AI.


  - ### Rust API

    - [uiuifree/rust-openai-chatgpt-api](https://github.com/uiuifree/rust-openai-chatgpt-api) <img src="https://img.shields.io/github/stars/uiuifree/rust-openai-chatgpt-api?style=social"/> : "rust-openai-chatgpt-api" is a Rust library for accessing the ChatGPT API, a powerful NLP platform by OpenAI. The library provides a simple and efficient interface for sending requests and receiving responses, including chat. It uses reqwest and serde for HTTP requests and JSON serialization. 



  - ### Csharp API

    - [betalgo/openai](https://github.com/betalgo/openai) <img src="https://img.shields.io/github/stars/betalgo/openai?style=social"/> : OpenAI ChatGPT, Whisper, GPT-3 , GPT-4, Azure OpenAI and DALL-E dotnet SDK. [betalgo.github.io/openai/](https://betalgo.github.io/openai/)

    - [OkGoDoIt/OpenAI-API-dotnet](https://github.com/OkGoDoIt/OpenAI-API-dotnet) <img src="https://img.shields.io/github/stars/OkGoDoIt/OpenAI-API-dotnet?style=social"/> : An unofficial C#/.NET SDK for accessing the OpenAI GPT-3 API. [www.nuget.org/packages/OpenAI/](https://www.nuget.org/packages/OpenAI/)

    - [RageAgainstThePixel/OpenAI-DotNet](https://github.com/RageAgainstThePixel/OpenAI-DotNet) <img src="https://img.shields.io/github/stars/RageAgainstThePixel/OpenAI-DotNet?style=social"/> : A Non-Official OpenAI RESTful API Client for dotnet.

    - [PawanOsman/ChatGPT.Net](https://github.com/PawanOsman/ChatGPT.Net) <img src="https://img.shields.io/github/stars/PawanOsman/ChatGPT.Net?style=social"/> : C# library for ChatGPT using official OpenAI API. [www.nuget.org/packages/ChatGPT.Net](https://www.nuget.org/packages/ChatGPT.Net)

    - [marcominerva/ChatGptNet](https://github.com/marcominerva/ChatGptNet) <img src="https://img.shields.io/github/stars/marcominerva/ChatGptNet?style=social"/> : A ChatGPT integration library for .NET.



  - ### Node.js API
    - [transitive-bullshit/chatgpt-api](https://github.com/transitive-bullshit/chatgpt-api) <img src="https://img.shields.io/github/stars/transitive-bullshit/chatgpt-api?style=social"/> : Node.js client for the unofficial ChatGPT API. 🔥




## Device Deployment

  - [MLC LLM](https://github.com/mlc-ai/mlc-llm) <img src="https://img.shields.io/github/stars/mlc-ai/mlc-llm?style=social"/> : Enable everyone to develop, optimize and deploy AI models natively on everyone's devices. [mlc.ai/mlc-llm](https://mlc.ai/mlc-llm/)

  - [Lamini](https://github.com/lamini-ai/lamini) <img src="https://img.shields.io/github/stars/lamini-ai/lamini?style=social"/> : Lamini: The LLM engine for rapidly customizing models 🦙.





## Applications


  - ### IDE

    - [Cursor](https://github.com/getcursor/cursor) <img src="https://img.shields.io/github/stars/getcursor/cursor?style=social"/> : An editor made for programming with AI 🤖. Long term, our plan is to build Cursor into the world's most productive development environment. [cursor.so](https://www.cursor.so/)



  - ### Academic

    - [binary-husky/chatgpt_academic](https://github.com/binary-husky/chatgpt_academic) <img src="https://img.shields.io/github/stars/binary-husky/chatgpt_academic?style=social"/> : ChatGPT 学术优化。科研工作专用ChatGPT拓展，特别优化学术Paper润色体验，支持自定义快捷按钮，支持markdown表格显示，Tex公式双显示，代码显示功能完善，新增本地Python工程剖析功能/自我剖析功能。

    - [kaixindelele/ChatPaper](https://github.com/kaixindelele/ChatPaper) <img src="https://img.shields.io/github/stars/kaixindelele/ChatPaper?style=social"/> : Use ChatGPT to summarize the arXiv papers. 全流程加速科研，利用chatgpt进行论文总结+润色+审稿+审稿回复。 💥💥💥面向全球，服务万千科研人的ChatPaper免费网页版正式上线：[https://chatpaper.org/](https://chatpaper.org/) 💥💥💥

    - [WangRongsheng/ChatGenTitle](https://github.com/WangRongsheng/ChatGenTitle) <img src="https://img.shields.io/github/stars/WangRongsheng/ChatGenTitle?style=social"/> : 🌟 ChatGenTitle：使用百万arXiv论文信息在LLaMA模型上进行微调的论文题目生成模型。

    - [nishiwen1214/ChatReviewer](https://github.com/nishiwen1214/ChatReviewer) <img src="https://img.shields.io/github/stars/nishiwen1214/ChatReviewer?style=social"/> : ChatReviewer: use ChatGPT to review papers; ChatResponse: use ChatGPT to respond to reviewers. 💥💥💥ChatReviewer的第一版网页出来了！！！ 直接点击：[https://huggingface.co/spaces/ShiwenNi/ChatReviewer](https://huggingface.co/spaces/ShiwenNi/ChatReviewer)

    - [Shiling42/web-simulator-by-GPT4](https://github.com/Shiling42/web-simulator-by-GPT4) <img src="https://img.shields.io/github/stars/Shiling42/web-simulator-by-GPT4?style=social"/> : Online Interactive Physical Simulation Generated by GPT-4. [shilingliang.com/web-simulator-by-GPT4/](https://shilingliang.com/web-simulator-by-GPT4/)





  - ### Wechat

    - [fuergaosi233/wechat-chatgpt](https://github.com/fuergaosi233/wechat-chatgpt) <img src="https://img.shields.io/github/stars/fuergaosi233/wechat-chatgpt?style=social"/> : Use ChatGPT On Wechat via wechaty.

    - [formulahendry/my-wechat-bot](https://github.com/formulahendry/my-wechat-bot) <img src="https://img.shields.io/github/stars/formulahendry/my-wechat-bot?style=social"/> : 基于 ChatGPT API 和 Wechaty，搭建 ChatGPT 微信机器人。


  - ### Translator

    - [yetone/openai-translator](https://github.com/yetone/openai-translator) <img src="https://img.shields.io/github/stars/yetone/openai-translator?style=social"/> : The translator that does more than just translation - powered by OpenAI.



  - ### GUI

    - [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) <img src="https://img.shields.io/github/stars/AUTOMATIC1111/stable-diffusion-webui?style=social"/> : Stable Diffusion web UI. A browser interface based on Gradio library for Stable Diffusion.

    - [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) <img src="https://img.shields.io/github/stars/oobabooga/text-generation-webui?style=social"/> : Text generation web UI. A gradio web UI for running Large Language Models like LLaMA, llama.cpp, GPT-J, Pythia, OPT, and GALACTICA.

    - [lencx/ChatGPT](https://github.com/lencx/ChatGPT) <img src="https://img.shields.io/github/stars/lencx/ChatGPT?style=social"/> : 🔮 ChatGPT Desktop Application (Mac, Windows and Linux). [NoFWL](https://app.nofwl.com/).

    - [ChatGPT-Desktop](https://github.com/ChatGPT-Desktop/ChatGPT-Desktop) <img src="https://img.shields.io/github/stars/ChatGPT-Desktop/ChatGPT-Desktop?style=social"/> : ChatGPT 跨平台客户端，快捷键快速唤醒窗口，问答快人一步！ 基于 tauri + vue3 开发的跨平台桌面端应用。 

    - [sonnylazuardi/chat-ai-desktop](https://github.com/sonnylazuardi/chat-ai-desktop) <img src="https://img.shields.io/github/stars/sonnylazuardi/chat-ai-desktop?style=social"/> : Chat AI Desktop App. Unofficial ChatGPT desktop app for Mac & Windows menubar using Tauri & Rust.

    - [Yidadaa/ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web) <img src="https://img.shields.io/github/stars/Yidadaa/ChatGPT-Next-Web?style=social"/> : 一键拥有你自己的 ChatGPT 网页服务。 One-Click to deploy your own ChatGPT web UI. 

    - [202252197/ChatGPT_JCM](https://github.com/202252197/ChatGPT_JCM) <img src="https://img.shields.io/github/stars/202252197/ChatGPT_JCM?style=social"/> : OpenAI Manage Web. OpenAI管理界面，聚合了OpenAI的所有接口进行界面操作。

    - [m1guelpf/browser-agent](https://github.com/m1guelpf/browser-agent) <img src="https://img.shields.io/github/stars/m1guelpf/browser-agent?style=social"/> : A browser AI agent, using GPT-4. [docs.rs/browser-agent](https://docs.rs/browser-agent/latest/browser_agent/)

    - [sigoden/aichat](https://github.com/sigoden/aichat) <img src="https://img.shields.io/github/stars/sigoden/aichat?style=social"/> : Using ChatGPT/GPT-3.5/GPT-4 in the terminal. 

    - [wieslawsoltes/ChatGPT](https://github.com/wieslawsoltes/ChatGPT) <img src="https://img.shields.io/github/stars/wieslawsoltes/ChatGPT?style=social"/> : A ChatGPT C# client for graphical user interface runs on MacOS, Windows, Linux, Android, iOS and Browser. Powered by [Avalonia UI](https://www.avaloniaui.net/) framework. [wieslawsoltes.github.io/ChatGPT/](https://wieslawsoltes.github.io/ChatGPT/)

    - [sigoden/aichat](https://github.com/GaiZhenbiao/ChuanhuChatGPT) <img src="https://img.shields.io/github/stars/GaiZhenbiao/ChuanhuChatGPT?style=social"/> : GUI for ChatGPT API and any LLM. 川虎 Chat 🐯 Chuanhu Chat. 为ChatGPT/ChatGLM/LLaMA/StableLM/MOSS等多种LLM提供了一个轻快好用的Web图形界。



## Blogs

  - [知乎「猛猿」](https://www.zhihu.com/people/lemonround)
    - [2023-02-25，ChatGPT技术解析系列之：GPT1、GPT2与GPT3](https://zhuanlan.zhihu.com/p/609367098?utm_id=0)    
  - 微信公众号「微软科技」
    - [2023-02-16，揭秘ChatGPT走红背后的独门云科技！](https://mp.weixin.qq.com/s/qYZ7G5uLHTiLG8AonIch8g)
  - 微信公众号「Azure云科技」
    - [2023-02-15，微软 Azure 作为 OpenAI 独家云服务提供商，助力企业致胜人工智能时代](https://mp.weixin.qq.com/s/SCmWX4uz3Ici2Shy6r1x7Q)
  - 微信公众号「量子位」
    - [2023-02-05，教ChatGPT学会看图的方法来了](https://mp.weixin.qq.com/s/OyLnRKgsklzQ09y9irtdQg)
    - [2023-02-12，ChatGPT背后模型被证实具有人类心智！斯坦福新研究炸了，知名学者：“这一天终于来了”](https://mp.weixin.qq.com/s/zgrJVFvkqG69BrQCky193A)
    - [2023-02-13，让ChatGPT长“手”！Meta爆火新论文，让语言模型学会自主使用工具](https://mp.weixin.qq.com/s/nca9jMOXgMKfhA8bo0FQvw)
    - [2023-02-15，ChatGPT低成本复现流程开源！任意单张消费级显卡可体验，显存需求低至1.62GB](https://mp.weixin.qq.com/s/GcqFifmpE3_VvuAcJPsf-A)
    - [2023-03-15，GPT-4发布！ChatGPT大升级！太太太太强了！](https://mp.weixin.qq.com/s/6u33Xnp4oEHq26WR4W1kdg)
    - [2023-03-15，微软为ChatGPT打造专用超算！砸下几亿美元，上万张英伟达A100打造](https://mp.weixin.qq.com/s/jae8CoMWMKqLVhApqBcTfg)
  - 微信公众号「机器之心」
    - [2023-02-15，开源方案复现ChatGPT流程！1.62GB显存即可体验，单机训练提速7.73倍](https://mp.weixin.qq.com/s/j8gvD_4ViRE4WQaQlcnmrQ)
    - [2023-02-19，跟李沐学ChatGPT背后技术：67分钟读透InstructGPT论文](https://mp.weixin.qq.com/s/s5WrGn_dQyHrsZP8qsI2ag)
    - [2023-02-21，复旦发布中国版ChatGPT：MOSS开启测试冲上热搜，服务器挤爆](https://mp.weixin.qq.com/s/LjwSozikB6CK5zh2Nd2JHw)
    - [2023-03-13，清华朱军团队开源首个基于Transformer的多模态扩散大模型，文图互生、改写全拿下](https://mp.weixin.qq.com/s/B68hXlFxA9L5jiWiMrEEiA)
    - [2023-03-14，真·ChatGPT平替：无需显卡，MacBook、树莓派就能运行LLaMA](https://mp.weixin.qq.com/s/7bRwX047jkZC53KYbhKARw)
    - [2023-03-15，GPT-4震撼发布：多模态大模型，直接升级ChatGPT、必应，开放API，游戏终结了？](https://mp.weixin.qq.com/s/kA7FBZsT6SIvwIkRwFS-xw)
    - [2023-04-02，3090单卡5小时，每个人都能训练专属ChatGPT，港科大开源LMFlow](https://mp.weixin.qq.com/s/LCGQyNA6sHcdfIIARSNlww)
    - [2023-04-06，CV不存在了？Meta发布「分割一切」AI 模型，CV或迎来GPT-3时刻](https://mp.weixin.qq.com/s/-LWG3rOz60VWiwdYG3iaWQ)
  - 微信公众号「图灵人工智能」
    - [2023-02-04，盖茨盛赞ChatGPT：人工智能历史意义不亚于“PC或互联网诞生”](https://mp.weixin.qq.com/s/51v_fUjQe3EewwOIxlLghw)
    - [2023-02-06，ChatGPT专题|ChatGPT之父传奇：8岁会编程，16岁出柜，2个月做到月活过亿](https://mp.weixin.qq.com/s/jodwa-a644vECTnrRqCuAA)
    - [2023-02-08，ChatGPT专题|为什么ChatGPT这么强？—— 一文读懂ChatGPT原理！](https://mp.weixin.qq.com/s/QNuKQ2Mgfn5K22JuUe2dHA)
    - [2023-02-11，ChatGPT专题|万字拆解！追溯ChatGPT各项能力的起源](https://mp.weixin.qq.com/s/4l0ADjdsCxSVvBeVKxSqWA)
    - [2023-02-15，ChatGPT专题|ChatGPT是第一个真正意义的人工通用智能](https://mp.weixin.qq.com/s/V7gptx740dDtVyQAgdhnqA)
    - [2023-02-16，ChatGPT专题|ChatGPT 算法原理](https://mp.weixin.qq.com/s/aIzwuATN71etbUrrQWYOkA)
    - [2023-02-16，ChatGPT专题|由ChatGPT反思大语言模型（LLM）的技术精要](https://mp.weixin.qq.com/s/SthaVFuAzvPnpCVdwaZYdA)
    - [2023-02-17，ChatGPT专题|ChatGPT背后真正的英雄：OpenAI首席科学家Ilya Sutskever的信仰之跃](https://mp.weixin.qq.com/s/EnRAcqiugR_xr7Mn0WJXLA)
    - [2023-02-18，ChatGPT专题|ChatGPT学了一门清华慕课，还拿到了课程证书](https://mp.weixin.qq.com/s/enaw41QEyiJ0ecNmjyEctw)
    - [2023-02-18，ChatGPT专题|关于GPT，人工智能，以及人的一些思考](https://mp.weixin.qq.com/s/SBpnmsc11C4fcH5xeftQdQ)
    - [2023-02-19，ChatGPT 专题：万字长文解释 ChatGPT 在做什么，以及为什么它能发挥作用？](https://mp.weixin.qq.com/s/gt0YxLG9ZW2wIg5rzfBhKw)
  - 微信公众号「硅星人」
    - [2022-12-03，行走的代码生成器：chatGPT要让谷歌和程序员“下岗”了](https://mp.weixin.qq.com/s/DXzZ_5RrRbVe5bWkpwFV6Q)
    - [2023-01-18，微软下个十年的想象力，藏在ChatGPT里](https://mp.weixin.qq.com/s/xjNipZ77I3eKbeYU5ZztZQ)
    - [2023-01-28，ChatGPT又赢了：带动股价涨三倍，成考试神器](https://mp.weixin.qq.com/s/BCfI_IhbIvLaAphYheM7yQ)
    - [2023-02-07，搜索大变天！谷歌推出Bard对抗ChatGPT，打响保卫战](https://mp.weixin.qq.com/s/33-Cg7Vn3Pmzuv_2IMHLzg)
    - [2023-02-16，谁拖了中国ChatGPT的后腿？](https://mp.weixin.qq.com/s/66ILghJKHjQhEVJ3r1xi7A)
    - [2023-02-24，OpenAI造就硅谷新“黑帮”：ChatGPT爆火背后的神秘大佬、技术版图和资本故事](https://mp.weixin.qq.com/s/eMwGbvxE_pCr1r1k18_yrA)
    - [2023-02-25，英伟达再度站上风口](https://mp.weixin.qq.com/s/_OM1_Pf1GLHW3zuF-3F93Q)
    - [2023-03-03，ChatGPT的战争不会浓缩于一个晚上](https://mp.weixin.qq.com/s/GJ94vpO9sRrXttdBo9oD2w)
    - [2023-03-15，OpenAI发布GPT-4：能识图能算税，ChatGPT摆脱Chat，再次进化](https://mp.weixin.qq.com/s/JahPijUPjxrzuLhq0esIUg)
    - [2023-03-17，GPT-4撑腰，Office全家桶集体升级，微软向谷歌丢出“王炸”](https://mp.weixin.qq.com/s/Ef_4FesHTP83NjZ3Knu5pA)
    - [2023-03-22，对抗ChatGPT，谷歌Bard公测炸场了：巨头开启AI对决](https://mp.weixin.qq.com/s/TkfaTNFz4bM6EnHygNymqw)
    - [2023-03-25，联网之后的ChatGPT，已经远不止“iPhone时刻”那么简单](https://mp.weixin.qq.com/s/_vn4RAqtRaNlBNP9W1sQcA)
    - [2023-03-30，AI太强，人类危险？马斯克、图灵奖得主紧急呼吁暂停GPT-4模型后续研发](https://mp.weixin.qq.com/s/QrrVefyvrOQ8IbAVzWA-6w)
    - [2023-04-01，OpenAI秘史公开：马斯克和奥特曼的战争，与钱无关](https://mp.weixin.qq.com/s/h_juJuhjVt8z-uu4qjaUFw)
    - [2023-04-05，这些让人惊呼好用的神器背后，原来都是ChatGPT](https://mp.weixin.qq.com/s/KL6OFAhPfr_OC80I_W6b3g)
    - [2023-04-07，Meta新模型“分割一切”：抠图完成究极进化，计算机视觉迎来GPT-3时刻](https://mp.weixin.qq.com/s/UUSmg6M5F6FJDs2i_-98dQ)
  - 微信公众号「通用人工智能联盟」
    - [2023-01-31，通用人工智能技术综述（一）](https://mp.weixin.qq.com/s/s1A0dHDs0ptNLIKXNivB8g)
    - [2023-02-01，通用人工智能技术综述（二）](https://mp.weixin.qq.com/s/dBAHHdcQPbogxyOv-yTvzg)
    - [2023-02-02，通用人工智能综述（三）](https://mp.weixin.qq.com/s/PjUPumRc9fFCmien71odsw)
    - [2023-02-04，通用人工智能技术综述（四）](https://mp.weixin.qq.com/s/3w-T6V9h3zgJUFxb2D7FXQ)
    - [2023-02-08，通用人工智能技术综述（五）](https://mp.weixin.qq.com/s/Bz4-AQ6UcFKTCSKoDwUrcg)
    - [2023-02-12，ChatGPT的开发及部署成本略析](https://mp.weixin.qq.com/s/cqfUl2lBGhWtVj6NbWbuew)
  - 微信公众号「计算机视觉研究院」
    - [2023-02-09，计算机视觉研究院亲自体验ChatGPT的感受，太疯狂了！](https://mp.weixin.qq.com/s/82Z3cODnPbwpStXIhnuJyw)
    - [2023-02-16，Image GPT——手把手教你搭建](https://mp.weixin.qq.com/s/gH_K_9Qo67HoNnSOnBevqw)
    - [2023-02-20，7 Papers | 超越GPT 3.5的小模型；对ChatGPT摸底考试](https://mp.weixin.qq.com/s/_HV9atcakv0sWD5X4tloPw)
  - 微信公众号「江大白」
    - [2023-02-15，万字拆解，ChatGPT各项能力的起源追溯](https://mp.weixin.qq.com/s/l0uGPO4vdFQzwCSP-HQQgg)
    - [2023-03-02，ChatGPT团队背景研究报告，大厂不再是顶尖人才第一选择！](https://mp.weixin.qq.com/s/F_9fChIMkuZLoUfhnenwAw)
    - [2023-03-03，行业热点 | ChatGPT数据集深度解密](https://mp.weixin.qq.com/s/mQiZIf-1QolCkX-2jTUa5Q)
    - [2023-03-13，北大团队搞出ChatExcel，说人话自动处理表格，免费且不限次使用！](https://mp.weixin.qq.com/s/H8aG9AewM0npJCpA2A0YGQ)
    - [2023-03-23，脑洞大开，如何利用ChatGPT搞科研？](https://mp.weixin.qq.com/s/HZvUfwpmPQC6OOX2Qyr-JQ)
    - [2023-03-29，GPT-4 的独立创业之路，一个人就是一家公司！](https://mp.weixin.qq.com/s/Qu-OXSoDS5hmdPe6EENM4w)
    - [2023-03-30，开源版ChatGPT项目，30分钟训完，性能堪比GPT3.5！（附源码）](https://mp.weixin.qq.com/s/x-UYyeAQc8NF2TiW8XLJHg)
    - [2023-04-03，学术版专用Chatgpt火热开源，科研工作必备，附源码！](https://mp.weixin.qq.com/s/19jGbV37DhkihhKAxqBk7w)
    - [2023-04-14，阿里版GPT通义千问实测来了！数学、编程、情书全套整活](https://mp.weixin.qq.com/s/a5NRdeR703CVBsG9xYgUlA)
  - 微信公众号「WeThinkln」
    - [2023-02-12，Rocky和ChatGPT“谈笑风生”的日子 |【AI行研&商业价值分析】](https://mp.weixin.qq.com/s/rV6J6UZgsJT-4HI49GBBaw)
    - [2023-02-26，深入浅出解析ChatGPT引领的科技浪潮 |【AI行研&商业价值分析】](https://mp.weixin.qq.com/s/FLLtb_9shzFmH1wpV7oP_Q)
  - 微信公众号「所向披靡的张大刀」
    - [2023-04-07，分割大一统——Segment Anything深度体验](https://mp.weixin.qq.com/s/qtk1Ds3hdNi4NOwrw2tDrg)
  - 微信公众号「算法邦」
    - [2023-03-06，没有这些，别妄谈做ChatGPT了](https://mp.weixin.qq.com/s/BwFUYFbkvAdDRE1Zqt_Qcg)
    - [2023-03-29，GPT-4将如何冲击计算机视觉领域？](https://mp.weixin.qq.com/s/KIFb24nxEvxIlyG23sy8bQ)
    - [2023-04-01，GPT-4的前世、今生和未来！](https://mp.weixin.qq.com/s/QNSbLdj5MdHuatdxW74QPQ)
    - [2023-04-03，ChatGPT成功背后的秘密，开源了！](https://mp.weixin.qq.com/s/V6Qgdf6JzfT7KGWVgNqWsQ)
    - [2023-04-05，如何与ChatGPT4结对编程提升研发效率](https://mp.weixin.qq.com/s/UJgNjIdQ13SuGHy2p7XE0Q)
  - 微信公众号「极市平台」
    - [2023-03-28，GPT系列来龙去脉大起底（一）｜第一代 GPT：无标注数据预训练生成式语言模型](https://mp.weixin.qq.com/s/wzZOjBJYtBpVZB-PzZenmQ)
    - [2023-04-06，GPT系列来龙去脉大起底（一）｜GPT-2：GPT 在零样本多任务学习的探索](https://mp.weixin.qq.com/s/YekKHeJD0KcCJ_73Wriuqw)
    - [2023-04-06，压缩下一个 token 通向超过人类的智能](https://mp.weixin.qq.com/s/UCB9-XPxZ0UA-kifakudFQ)
  - 微信公众号「计算机视觉与机器学习」
    - [2023-04-06，不止 GPT4 ，大语言模型的演变之路！](https://mp.weixin.qq.com/s/YhvtxqBszvfcmtLvZgWqhw)
    - [2023-04-04，GPT-4 版“贾维斯”诞生，国外小哥用它 4 分钟创建网站、聊天就能创建 GitHub repo......](https://mp.weixin.qq.com/s/agtQeScBNBvSX1yqLTW4JQ)
    - [2023-04-03，CVPR 2023 | 模块化MoE将成为视觉多任务学习基础模型](https://mp.weixin.qq.com/s/VsGOio9mn-o82bWI1MMUcA)
  - 微信公众号「CV技术指南」
    - [2023-04-07，3090单卡5小时，每个人都能训练专属ChatGPT，港科大开源LMFlow](https://mp.weixin.qq.com/s/h6zbAVgFpW0ccdEHjLFpdQ)
    - [2023-04-07，上线一天，4k star | Facebook：Segment Anything](https://mp.weixin.qq.com/s/G7xeuZE3vHuujQrDxIrePA)
  - 微信公众号「计算机视觉工坊」
    - [2023-04-07，超震撼！Meta发布「分割一切」AI 模型！](https://mp.weixin.qq.com/s/_IbadabLJnvv1_a-NsAJfg)
    - [2023-04-08，CV开启大模型时代！谷歌发布史上最大ViT：220亿参数，视觉感知力直逼人类](https://mp.weixin.qq.com/s/ur2WTw95pUduxh9EYULR_Q)
  - 微信公众号「新智元」
    - [2023-02-03，60天月活破亿，ChatGPT之父传奇：16岁出柜，20岁和男友一同当上CEO](https://mp.weixin.qq.com/s/W1xfLgZXWL3lfP4_54SQKw)
    - [2023-03-17，微软深夜放炸弹！GPT-4 Office全家桶发布，10亿打工人被革命](https://mp.weixin.qq.com/s/YgiurOE0uZ7lRDx1ehpbhQ)
  - 微信公众号「智东西」
    - [2023-02-06，ChatGPT版搜索引擎突然上线，科技巨头们坐不住了！](https://mp.weixin.qq.com/s/lncJm6hmK3AQNF2paWI5Dw)
    - [2023-04-07，ChatGPT和Matter两大风口汇合！AWE同期AIoT智能家居峰会月底举行，首批嘉宾公布](https://mp.weixin.qq.com/s/cuI8sSff_zGiLtwukAcLRw)
    - [2023-04-23，BroadLink CEO刘宗孺：ChatGPT助推全屋智能管家式变革](https://mp.weixin.qq.com/s/t4BPrvYT8oF8lGKutjpJtQ)
    - [2023-04-23，复旦MOSS升级版开源上线；马斯克启动TruthGPT；海康训练出百亿参数CV大模型丨AIGC大事周报](https://mp.weixin.qq.com/s/gBDcHw1SFSCWpJIxeC5vHg)
  - 微信公众号「CSDN」
    - [2023-03-25，ChatGPT 已成为下一代的新操作系统！](https://mp.weixin.qq.com/s/MwrMhVydbhpP6c0AvPp8oQ)
    - [2023-04-06，CV 迎来 GPT-3 时刻，Meta 开源万物可分割 AI 模型和 1100 万张照片，1B+掩码数据集！](https://mp.weixin.qq.com/s/spBwU0UecbxbEl88SA4GJQ)
  - 微信公众号「刘润」
    - [2023-02-08，ChatGPT：一个人不管有多大的梦想，还是要有盖世武功](https://mp.weixin.qq.com/s/Dd28kONcjwiBYPuDUD8R7g)
    - [2023-02-09，ChatGPT：你来了，那我怎么办？](https://mp.weixin.qq.com/s/3wikMRAJqZtWHaC5dUVgbQ)
    - [2023-02-12，ChatGPT引爆新一轮科技军备赛](https://mp.weixin.qq.com/s/4oofzJywBsG9SF6Hb48WNQ)
    - [2023-02-14，ChatGPT创始人，给我们上的8堂课](https://mp.weixin.qq.com/s/js-fY2nJBAr_pZItTw-PMg)
  - 微信公众号「AI算法与图像处理」
    - [2023-02-16，推荐一个方便好用的 ChatGPT 客户端！](https://mp.weixin.qq.com/s/Lu0WqBxRcACfucgmTk2OEw)
  - 微信公众号「中国图象图形学报」
    - [2023-02-16，编委动态 | 浅析ChatGPT：历史沿革、应用现状及前景展望](https://mp.weixin.qq.com/s/EgiBEb7D4HkaKtjmsMnRHA)
  - 微信公众号「脑机接口社区」
    - [2023-02-15，ChatGPT发展历程、原理、技术架构详解和产业未来](https://mp.weixin.qq.com/s/LhcqK6W7OTB0Y1LfZIsGfA)
  - 微信公众号「中国科学院自动化研究所」
    - [2023-02-15，嗨ChatGPT，人类对你最好奇的是什么呢？这篇文章一一解答！丨智言智语](https://mp.weixin.qq.com/s/BYCemIdTx2kZ9jotF13u2w)
  - 微信公众号「玩转VS Code」
    - [2023-02-16，目前最火的 ChatGPT 开源项目！](https://mp.weixin.qq.com/s/E2-MrsKfvNxIvuW7h4NT6Q)
  - 微信公众号「人工智能学家」
    - [2023-02-15，人机交互新时代：多维度快速看清ChatGPT（附下载）](https://mp.weixin.qq.com/s/MHqn53ZFjXPt8tC1d9oCOA)
  - 微信公众号「新机器视觉」
    - [2023-02-13，ChatGPT 算法原理](https://mp.weixin.qq.com/s/DYRjmJ7ePTqV1RFkBZFCTw)
  - 微信公众号「投行圈子」
    - [2023-02-11，ChatGPT研究框架（80页PPT）](https://mp.weixin.qq.com/s/eGLqpTvFztok3MWE3ISc2A)
  - 微信公众号「机器学习算法那些事」
    - [2023-02-08，多模态版ChatGPT，拿下视觉语言新SOTA， 代码已开源](https://mp.weixin.qq.com/s/lsRSzwsLiTo6anPnKFa-4A)
  - 微信公众号「机器学习算法工程师」
    - [2023-04-08，CV突然进入GPT4时代！Meta和智源研究院发布「分割一切」AI 模型](https://mp.weixin.qq.com/s/9zTX0awkGPc9kfoX2QpDIg)
  - 微信公众号「人工智能与算法学习」
    - [2023-02-15，ChatGPT数据集之谜](https://mp.weixin.qq.com/s/CFgsiJ7a2mXQNAWkQxScYQ)
    - [2023-03-10，王炸！微软发布Visual ChatGPT：视觉模型加持ChatGPT实现丝滑聊天](https://mp.weixin.qq.com/s/jQd0xujid66CrcBrhhZoLQ)
  - 微信公众号「量子学派」
    - [2023-02-10，封杀这个公式，ChatGPT智商将为零](https://mp.weixin.qq.com/s/l1Qxe3rGTYuIumHq02exsg)
    - [2023-02-10，ChatGPT，一种更中心化的权力？](https://mp.weixin.qq.com/s/-qmccVnv_rpKVdFP6x4GNg)
  - 微信公众号「42章经」
    - [2023-02-13，我是怎样用一周时间研究 ChatGPT 的？](https://mp.weixin.qq.com/s/obVI3ENpMgaq4AKZs6Hw1w)
  - 微信公众号「人工智能技术与咨询」
    - [ChatGPT四大应用主线及相关细分场景](https://mp.weixin.qq.com/s/f8cmRVs0ys7FNyNU1qbP6g)
  - 微信公众号「应用语言学研习」
    - [2023-02-17，如何利用ChatGPT搞科研？](https://mp.weixin.qq.com/s/sW_utRBS_jJAaWfGo_eT5g)
  - 微信公众号「机器之能」
    - [2023-03-22，比尔·盖茨：AI时代已经开启，GPT是40年来最具革命性技术](https://mp.weixin.qq.com/s/j3D7g_1HeKZbznOqqU2pxw)
  - 微信公众号「机器学习研究组订阅」
    - [2023-03-26，震惊科学界！微软154页研究刷屏：GPT-4能力接近人类，「天网」初现？](https://mp.weixin.qq.com/s/C0qwDb_ASCbmP8sHgH97Jg)
  - 微信公众号「浮之静」
    - [2022-12-14，流量密码：ChatGPT 开源的一些思考](https://mp.weixin.qq.com/s/-lpQycfKVQ1gLKjoMrTvpA)
    - [2023-02-08，ChatGPT 扫盲指南](https://mp.weixin.qq.com/s/4RczQBdAmnYSdlhMBcXcZA)
    - [2023-03-01，一文读懂 OpenAI](https://mp.weixin.qq.com/s/_ovmBsJ7EQr_k4JnSKtuLw)
    - [2023-03-15，AI 里程碑：GPT-4 发布了！](https://mp.weixin.qq.com/s/n8ttVSJmd44sBdpnL3Whxw)
    - [2023-03-27，AI 浪潮下的一些浅思](https://mp.weixin.qq.com/s/1TYrtufxtLcMy0RolNAbhg)
  - 微信公众号「北京日报」
    - [2023-02-13，正式发布！北京：支持头部企业打造对标ChatGPT的大模型](https://mp.weixin.qq.com/s/gDRibma_0wC8ZyEXwTkvgQ)
  - 微信公众号「学术头条」
    - [2023-02-22，揭秘ChatGPT背后的AI“梦之队”：90后科研“后浪”展示强大创新能力｜智谱研究报告](https://mp.weixin.qq.com/s/sncE01utzu_-r3dLFYU5QA)
  - 微信公众号「中国人工智能学会」
    - [2023-03-09，会员两会之声丨CAAI副理事长周志华委员：AI发展急需加强机器学习基础研究](https://mp.weixin.qq.com/s/zYElMLISwLxOUI6ZH_K-mg)
  - 微信公众号「人工智能研究」
    - [2023-03-11，哈工大NLP研究所ChatGPT调研报告发布！](https://mp.weixin.qq.com/s/u17VEv0VM8MXYyB7jcV-yA)
  - 微信公众号「OpenFPGA」
    - [2023-03-13，在FPGA设计中怎么应用ChatGPT？](https://mp.weixin.qq.com/s/BvCFoAi9tAvSs4QS4BFRdA)
    - [2023-03-27，ChatGPT推荐的开源项目，到底靠不靠谱？](https://mp.weixin.qq.com/s/_ERFebXaLUbF3EQs_ZyPIQ)
  - 微信公众号「AI科技评论」
    - [2023-03-14，何恺明 MIT 最新演讲：未来工作将聚焦 AI for science](https://mp.weixin.qq.com/s/8oiHz34DpfDJmT4IPzU8IA)
  - 微信公众号「HelloGitHub」
    - [2023-03-17，GPT-4 来了！这些开源的 GPT 应用又要变强了](https://mp.weixin.qq.com/s/MeexLX_aOyUKHtaiyuwMTA)
  - 微信公众号「脚本之家」
    - [2023-03-23，GPT-4 Copilot X震撼来袭！AI写代码效率10倍提升，码农遭降维打击](https://mp.weixin.qq.com/s/XCBPSCLSDUSiu3CP54PfWg)
  - 微信公众号「FightingCV」
    - [2023-03-23，OpenAI重磅研究：ChatGPT可能影响80%工作岗位，收入越高影响越大](https://mp.weixin.qq.com/s/DUiEqgz-Ytf6c8NU8f7O3w)
  - 微信公众号「科金中心」
    - [2023-03-22，今日关注 | 比尔盖茨：超级人工智能还没来 GPT模型是40余年来最革命性技术进步](https://mp.weixin.qq.com/s/vBkbE04Oz0ssYqjsvIacPg)
  - 微信公众号「findyi」
    - [2023-04-06，ChatGPT！王炸级更新！！！](https://mp.weixin.qq.com/s/F3gSN_GWvvCOR2zGva4Oew)
  - 微信公众号「AI能力站」
    - [2023-04-01，AIGC、ChatGPT和LLM三者之间的联系](https://mp.weixin.qq.com/s/O-A3uU1g8_LkOO1VhxYX4Q)
  - 微信公众号「孙立平社会观察」
    - [2023-04-07，霍金：失控的人工智能很难被阻止住](https://mp.weixin.qq.com/s/Zd4o3p4ysTJ7_kNzGivKPA)
  - 微信公众号「世界经济论坛」
    - [2023-04-01，比尔·盖茨：人工智能变革前夜的展望](https://mp.weixin.qq.com/s/O-AUjuVgfcDk2OrxBOcL_g)
  - 微信公众号「新华数字」
    - [2022-12-06，AIGC：ChatGPT的未来展望](https://mp.weixin.qq.com/s/sZUwvE6kehkTuZ1wuXzn2g)
  - 微信公众号「猫说AI」
    - [2023-04-04，ChatGPT开源平替--ChatGLM](https://mp.weixin.qq.com/s/sCTuMgbGK6N_bThOhJJ9-w)
  - 微信公众号「资本实验室」
    - [2023-02-13，ChatGPT爆火之下，生成式人工智能的「远忧近虑」| 海外周选](https://mp.weixin.qq.com/s/hrIwPA_eBu2sUmfW7mYlsw)
    - [2023-02-15，ChatGPT爆火之际，一文看清全球各方力量的应对与跟进行动](https://mp.weixin.qq.com/s/q-xuf3DUtsqW9U4SL5p18A)
  - 微信公众号「空中机器人前沿」
    - [2023-03-22，在「机器人领域」使用ChatGPT提高生产力](https://mp.weixin.qq.com/s/MB9pcqzLHb_oNNdDYa2oSA)
  - 微信公众号「CVHub」
    - [2023-04-06，《万字长文带你解读AIGC》系列之技术篇](https://mp.weixin.qq.com/s/6jMCd9yn_vBLiLJGBpSB2g)
  - 微信公众号「芯榜」
    - [2023-04-16，思特威：人工智能浪潮，将机器视觉冲向新蓝海](https://mp.weixin.qq.com/s/jtJvltmjSeCi47XiVOzzdw)
  - 微信公众号「数智前线」
    - [2023-04-12，阿里通义千问，通向企业](https://mp.weixin.qq.com/s/L3FCVJVbMdKdeP6m8B9Lmg)
    - [2023-04-18，解码商汤大模型体系](https://mp.weixin.qq.com/s/3mkYe-UAy3dJFMBbPvgbrA)
  - 微信公众号「智能进化论」
    - [2023-04-18，AI大模型内卷加剧，商汤凭什么卷进来](https://mp.weixin.qq.com/s/-az_NylC3EyqN4iYx8Sbrw)
  - 微信公众号「深蓝AI」
    - [2023-04-23，最新综述！AIGC到底是什么？都有哪些应用？一文尽览！](https://mp.weixin.qq.com/s/rp9XVUBrh17Wr57SPFgTvg)
  - 微信公众号「人工智能前沿讲习」
    - [2023-04-23，【综述专栏】“ChatGPT的问题、风险与机遇”会议综述](https://mp.weixin.qq.com/s/-Gi4xMUXYiI13DaTVgwUdQ)







