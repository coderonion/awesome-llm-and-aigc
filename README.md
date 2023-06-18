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
      - [Cpp Implementation](#cpp-implementation)
      - [Rust Implementation](#rust-implementation)
      - [zig Implementation](#zig-implementation)
    - [Awesome List](#awesome-list)
    - [Paper Overview](#paper-overview)
    - [Learning Resources](#learning-resources)
  - [Prompts](#prompts)
  - [Open API](#open-api)
    - [Python API](#python-api)
    - [Rust API](#rust-api)
    - [Csharp API](#csharp-api)
    - [Node.js API](#node.js-api)
  - [Datasets](#datasets)
    - [Multimodal Datasets](#multimodal-datasets)
  - [Applications](#applications)
    - [IDE](#ide)
    - [Wechat](#wechat)
    - [Translator](#translator)
    - [Local knowledge Base](#local-knowledge-base)
    - [Academic Field](#academic-field)
    - [Medical Field](#medical-field)
    - [Legal Field](#legal-field)
    - [Math Field](#math-field)
    - [Device Deployment](#device-deployment)
    - [GUI](#gui)
  - [Blogs](#blogs)



## Summary

  - ### Frameworks

    - #### Official Version

      - ##### Large Language Model
        ###### 大语言模型（LLM）

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

        - [StarCoder](https://github.com/bigcode-project/starcoder) <img src="https://img.shields.io/github/stars/bigcode-project/starcoder?style=social"/> : 💫 StarCoder is a language model (LM) trained on source code and natural language text. Its training data incorporates more that 80 different programming languages as well as text extracted from GitHub issues and commits and from notebooks. 

        - [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) <img src="https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=social"/> : Stanford Alpaca: An Instruction-following LLaMA Model.
        
        - [feizc/Visual-LLaMA](https://github.com/feizc/Visual-LLaMA) <img src="https://img.shields.io/github/stars/feizc/Visual-LLaMA?style=social"/> : Open LLaMA Eyes to See the World. This project aims to optimize LLaMA model for visual information understanding like GPT-4 and further explore the potentional of large language model.

        - [Lightning-AI/lightning-colossalai](https://github.com/Lightning-AI/lightning-colossalai) <img src="https://img.shields.io/github/stars/Lightning-AI/lightning-colossalai?style=social"/> : Efficient Large-Scale Distributed Training with [Colossal-AI](https://colossalai.org/) and [Lightning AI](https://lightning.ai/).

        - [GPT4All](https://github.com/nomic-ai/gpt4all) <img src="https://img.shields.io/github/stars/nomic-ai/gpt4all?style=social"/> : GPT4All: An ecosystem of open-source on-edge large language models. GTP4All is an ecosystem to train and deploy powerful and customized large language models that run locally on consumer grade CPUs.

        - [LangChain](https://github.com/hwchase17/langchain) <img src="https://img.shields.io/github/stars/hwchase17/langchain?style=social"/> :  ⚡ Building applications with LLMs through composability ⚡

        - [ChatALL](https://github.com/sunner/ChatALL) <img src="https://img.shields.io/github/stars/sunner/ChatALL?style=social"/> :  Concurrently chat with ChatGPT, Bing Chat, bard, Alpaca, Vincuna, Claude, ChatGLM, MOSS, iFlytek Spark, ERNIE and more, discover the best answers. [chatall.ai](http://chatall.ai/)

        - [1595901624/gpt-aggregated-edition](https://github.com/1595901624/gpt-aggregated-edition) <img src="https://img.shields.io/github/stars/1595901624/gpt-aggregated-edition?style=social"/> : 聚合ChatGPT官方版、ChatGPT免费版、文心一言、Poe、chatchat等多平台，支持自定义导入平台。

        - [FreedomIntelligence/LLMZoo](https://github.com/FreedomIntelligence/LLMZoo) <img src="https://img.shields.io/github/stars/FreedomIntelligence/LLMZoo?style=social"/> : ⚡LLM Zoo is a project that provides data, models, and evaluation benchmark for large language models.⚡ [Tech Report](https://github.com/FreedomIntelligence/LLMZoo/blob/main/assets/llmzoo.pdf)

        - [shm007g/LLaMA-Cult-and-More](https://github.com/shm007g/LLaMA-Cult-and-More) <img src="https://img.shields.io/github/stars/shm007g/LLaMA-Cult-and-More?style=social"/> : News about 🦙 Cult and other AIGC models.

        - [X-PLUG/mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) <img src="https://img.shields.io/github/stars/X-PLUG/mPLUG-Owl?style=social"/> : mPLUG-Owl🦉: Modularization Empowers Large Language Models with Multimodality.

        - [i-Code](https://github.com/microsoft/i-Code) <img src="https://img.shields.io/github/stars/microsoft/i-Code?style=social"/> : The ambition of the i-Code project is to build integrative and composable multimodal Artificial Intelligence. The "i" stands for integrative multimodal learning. "CoDi: Any-to-Any Generation via Composable Diffusion". (**[arXiv 2023](https://arxiv.org/abs/2305.11846)**). 

        - [FlagAI|悟道·天鹰（Aquila）](https://github.com/FlagAI-Open/FlagAI) <img src="https://img.shields.io/github/stars/FlagAI-Open/FlagAI?style=social"/> : FlagAI (Fast LArge-scale General AI models) is a fast, easy-to-use and extensible toolkit for large-scale model. Our goal is to support training, fine-tuning, and deployment of large-scale models on various downstream tasks with multi-modality.

        - [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) <img src="https://img.shields.io/github/stars/THUDM/ChatGLM-6B?style=social"/> : ChatGLM-6B: An Open Bilingual Dialogue Language Model | 开源双语对话语言模型。 ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。 "GLM: General Language Model Pretraining with Autoregressive Blank Infilling". (**[ACL 2022](https://aclanthology.org/2022.acl-long.26/)**).  "GLM-130B: An Open Bilingual Pre-trained Model". (**[ICLR 2023](https://openreview.net/forum?id=-Aw0rrrPUF)**). 

        - [Chinese LLaMA and Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) <img src="https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca?style=social"/> : 中文LLaMA&Alpaca大语言模型+本地CPU/GPU部署 (Chinese LLaMA & Alpaca LLMs). 本项目开源了中文LLaMA模型和指令精调的Alpaca大模型。这些模型在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力。"Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca". (**[arXiv 2023](https://arxiv.org/abs/2304.08177)**). 

        - [MOSS](https://github.com/OpenLMLab/MOSS) <img src="https://img.shields.io/github/stars/OpenLMLab/MOSS?style=social"/> : An open-source tool-augmented conversational language model from Fudan University. MOSS是一个支持中英双语和多种插件的开源对话语言模型，moss-moon系列模型具有160亿参数，在FP16精度下可在单张A100/A800或两张3090显卡运行，在INT4/8精度下可在单张3090显卡运行。MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。[txsun1997.github.io/blogs/moss.html](https://txsun1997.github.io/blogs/moss.html)

        - [CPM-Bee](https://github.com/OpenBMB/CPM-Bee) <img src="https://img.shields.io/github/stars/OpenBMB/CPM-Bee?style=social"/> : CPM-Bee是一个完全开源、允许商用的百亿参数中英文基座模型，也是[CPM-Live](https://live.openbmb.org/)训练的第二个里程碑。

        - [PandaLM](https://github.com/WeOpenML/PandaLM) <img src="https://img.shields.io/github/stars/WeOpenML/PandaLM?style=social"/> : PandaLM: Reproducible and Automated Language Model Assessment.

        - [SpeechGPT](https://github.com/0nutation/SpeechGPT) <img src="https://img.shields.io/github/stars/0nutation/SpeechGPT?style=social"/> : "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities". (**[arXiv 2023](https://arxiv.org/abs/2305.11000)**). 

        - [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese) <img src="https://img.shields.io/github/stars/Morizeyao/GPT2-Chinese?style=social"/> : Chinese version of GPT2 training code, using BERT tokenizer. 

        - [百度-文心大模型](https://wenxin.baidu.com/) : 百度全新一代知识增强大语言模型，文心大模型家族的新成员，能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。

        - [阿里云-通义千问](https://tongyi.aliyun.com/) : 通义千问，是阿里云推出的一个超大规模的语言模型，功能包括多轮对话、文案创作、逻辑推理、多模态理解、多语言支持。能够跟人类进行多轮的交互，也融入了多模态的知识理解，且有文案创作能力，能够续写小说，编写邮件等。

        - [商汤科技-日日新SenseNova](https://techday.sensetime.com/?utm_source=baidu-sem-pc&utm_medium=cpc&utm_campaign=PC-%E6%8A%80%E6%9C%AF%E4%BA%A4%E6%B5%81%E6%97%A5-%E4%BA%A7%E5%93%81%E8%AF%8D-%E6%97%A5%E6%97%A5%E6%96%B0&utm_content=%E6%97%A5%E6%97%A5%E6%96%B0&utm_term=%E6%97%A5%E6%97%A5%E6%96%B0SenseNova&e_creative=73937788324&e_keywordid=594802524403) : 日日新（SenseNova），是商汤科技宣布推出的大模型体系，包括自然语言处理模型“商量”（SenseChat）、文生图模型“秒画”和数字人视频生成平台“如影”（SenseAvatar）等。




      - ##### Vision Foundation Model
        ###### 视觉大模型（VFM）

        - [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) <img src="https://img.shields.io/github/stars/microsoft/visual-chatgpt?style=social"/> : Visual ChatGPT connects ChatGPT and a series of Visual Foundation Models to enable sending and receiving images during chatting. "Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models". (**[arXiv 2023](https://arxiv.org/abs/2303.04671)**). 

        - [InternImage](https://github.com/OpenGVLab/InternImage) <img src="https://img.shields.io/github/stars/OpenGVLab/InternImage?style=social"/> : "InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions". (**[CVPR 2023](https://arxiv.org/abs/2211.05778)**). 

        - [GLIP](https://github.com/microsoft/GLIP) <img src="https://img.shields.io/github/stars/microsoft/GLIP?style=social"/> : "Grounded Language-Image Pre-training". (**[CVPR 2022](https://arxiv.org/abs/2112.03857)**). 

        - [GLIPv2](https://github.com/microsoft/GLIP) <img src="https://img.shields.io/github/stars/microsoft/GLIP?style=social"/> : "GLIPv2: Unifying Localization and Vision-Language Understanding". (**[arXiv 2022](https://arxiv.org/abs/2206.05836)**). 

        - [DINO](https://github.com/IDEA-Research/DINO) <img src="https://img.shields.io/github/stars/IDEA-Research/DINO?style=social"/> : "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection". (**[ICLR 2023](https://arxiv.org/abs/2203.03605)**). 

        - [DINOv2](https://github.com/facebookresearch/dinov2) <img src="https://img.shields.io/github/stars/facebookresearch/dinov2?style=social"/> : "DINOv2: Learning Robust Visual Features without Supervision". (**[arXiv 2023](https://arxiv.org/abs/2304.07193)**). 

        - [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) <img src="https://img.shields.io/github/stars/IDEA-Research/GroundingDINO?style=social"/> : "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection". (**[arXiv 2023](https://arxiv.org/abs/2303.05499)**). "知乎「三分钟热度」《[十分钟解读Grounding DINO-根据文字提示检测任意目标](https://zhuanlan.zhihu.com/p/627646794)》"。

        - [SAM](https://github.com/facebookresearch/segment-anything) <img src="https://img.shields.io/github/stars/facebookresearch/segment-anything?style=social"/> : The repository provides code for running inference with the Segment Anything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. "Segment Anything". (**[arXiv 2023](https://arxiv.org/abs/2304.02643)**). 

        - [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) <img src="https://img.shields.io/github/stars/IDEA-Research/Grounded-Segment-Anything?style=social"/> : Marrying Grounding DINO with Segment Anything & Stable Diffusion & Tag2Text & BLIP & Whisper & ChatBot - Automatically Detect , Segment and Generate Anything with Image, Text, and Audio Inputs. We plan to create a very interesting demo by combining [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment Anything](https://github.com/facebookresearch/segment-anything) which aims to detect and segment Anything with text inputs!

        - [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) <img src="https://img.shields.io/github/stars/UX-Decoder/Segment-Everything-Everywhere-All-At-Once?style=social"/> : We introduce SEEM that can Segment Everything Everywhere with Multi-modal prompts all at once. SEEM allows users to easily segment an image using prompts of different types including visual prompts (points, marks, boxes, scribbles and image segments) and language prompts (text and audio), etc. It can also work with any combinations of prompts or generalize to custom prompts! "Segment Everything Everywhere All at Once". (**[arXiv 2023](https://arxiv.org/abs/2304.06718)**). 

        - [SAM3D](https://github.com/DYZhang09/SAM3D) <img src="https://img.shields.io/github/stars/DYZhang09/SAM3D?style=social"/> : "SAM3D: Zero-Shot 3D Object Detection via [Segment Anything](https://github.com/facebookresearch/segment-anything) Model". (**[arXiv 2023](https://arxiv.org/abs/2306.02245)**). 

        - [ImageBind](https://github.com/facebookresearch/ImageBind) <img src="https://img.shields.io/github/stars/facebookresearch/ImageBind?style=social"/> : "ImageBind: One Embedding Space To Bind Them All". (**[CVPR 2023](https://arxiv.org/abs/2305.05665)**).  

        - [Track-Anything](https://github.com/gaomingqi/Track-Anything) <img src="https://img.shields.io/github/stars/gaomingqi/Track-Anything?style=social"/> : Track-Anything is a flexible and interactive tool for video object tracking and segmentation, based on Segment Anything, XMem, and E2FGVI. "Track Anything: Segment Anything Meets Videos". (**[arXiv 2023](https://arxiv.org/abs/2304.11968)**). 

        - [qianqianwang68/omnimotion](https://github.com/qianqianwang68/omnimotion) <img src="https://img.shields.io/github/stars/qianqianwang68/omnimotion?style=social"/> : "Tracking Everything Everywhere All at Once". (**[arXiv 2023](https://arxiv.org/abs/2306.05422)**). 

        - [LLaVA](https://github.com/haotian-liu/LLaVA) <img src="https://img.shields.io/github/stars/haotian-liu/LLaVA?style=social"/> : 🌋 LLaVA: Large Language and Vision Assistant. Visual instruction tuning towards large language and vision models with GPT-4 level capabilities. [llava.hliu.cc](https://llava.hliu.cc/). "Visual Instruction Tuning". (**[arXiv 2023](https://arxiv.org/abs/2304.08485)**). 

        - [M3I-Pretraining](https://github.com/OpenGVLab/M3I-Pretraining) <img src="https://img.shields.io/github/stars/OpenGVLab/M3I-Pretraining?style=social"/> : "Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information". (**[arXiv 2022](https://arxiv.org/abs/2211.09807)**). 

        - [BEVFormer](https://github.com/fundamentalvision/BEVFormer) <img src="https://img.shields.io/github/stars/fundamentalvision/BEVFormer?style=social"/> : BEVFormer: a Cutting-edge Baseline for Camera-based Detection. "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers". (**[arXiv 2022](https://arxiv.org/abs/2203.17270)**). 

        - [Uni-Perceiver](https://github.com/fundamentalvision/Uni-Perceiver) <img src="https://img.shields.io/github/stars/fundamentalvision/Uni-Perceiver?style=social"/> : "Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks". (**[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Uni-Perceiver_Pre-Training_Unified_Architecture_for_Generic_Perception_for_Zero-Shot_and_CVPR_2022_paper.html)**). 

        - [AnyLabeling](https://github.com/vietanhdev/anylabeling) <img src="https://img.shields.io/github/stars/vietanhdev/anylabeling?style=social"/> : 🌟 AnyLabeling 🌟. Effortless data labeling with AI support from YOLO and Segment Anything! Effortless data labeling with AI support from YOLO and Segment Anything!

        - [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) <img src="https://img.shields.io/github/stars/CVHub520/X-AnyLabeling?style=social"/> : 💫 X-AnyLabeling 💫. Effortless data labeling with AI support from Segment Anything and other awesome models!

        - [Label Anything](https://github.com/open-mmlab/playground/tree/main/label_anything) <img src="https://img.shields.io/github/stars/open-mmlab/playground?style=social"/> : OpenMMLab PlayGround: Semi-Automated Annotation with Label-Studio and SAM. 







      - ##### AI Generated Content
        ###### 人工智能生成内容（AIGC）

        - [Stable Diffusion](https://github.com/CompVis/stable-diffusion) <img src="https://img.shields.io/github/stars/CompVis/stable-diffusion?style=social"/> : Stable Diffusion is a latent text-to-image diffusion model. Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work "High-Resolution Image Synthesis with Latent Diffusion Models". (**[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)**). 

        - [Stable Diffusion Version 2](https://github.com/Stability-AI/stablediffusion) <img src="https://img.shields.io/github/stars/Stability-AI/stablediffusion?style=social"/> : This repository contains [Stable Diffusion](https://github.com/CompVis/stable-diffusion) models trained from scratch and will be continuously updated with new checkpoints. "High-Resolution Image Synthesis with Latent Diffusion Models". (**[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)**). 

        - [StableStudio](https://github.com/Stability-AI/StableStudio) <img src="https://img.shields.io/github/stars/Stability-AI/StableStudio?style=social"/> : StableStudio by [Stability AI](https://stability.ai/). 👋 Welcome to the community repository for StableStudio, the open-source version of [DreamStudio](https://dreamstudio.ai/).

        - [DragGAN](https://github.com/XingangPan/DragGAN) <img src="https://img.shields.io/github/stars/XingangPan/DragGAN?style=social"/> : "Stable Diffusion Training with MosaicML. This repo contains code used to train your own Stable Diffusion model on your own data". (**[SIGGRAPH 2023](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)**).          
      
        - [AudioGPT](https://github.com/AIGC-Audio/AudioGPT) <img src="https://img.shields.io/github/stars/AIGC-Audio/AudioGPT?style=social"/> : AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head.

        - [PandasAI](https://github.com/gventuri/pandas-ai) <img src="https://img.shields.io/github/stars/gventuri/pandas-ai?style=social"/> : Pandas AI is a Python library that adds generative artificial intelligence capabilities to Pandas, the popular data analysis and manipulation tool. It is designed to be used in conjunction with Pandas, and is not a replacement for it.

        - [mosaicml/diffusion](https://github.com/mosaicml/diffusion) <img src="https://img.shields.io/github/stars/mosaicml/diffusion?style=social"/> : Stable Diffusion Training with MosaicML. This repo contains code used to train your own Stable Diffusion model on your own data.

        - [ControlNet](https://github.com/lllyasviel/ControlNet) <img src="https://img.shields.io/github/stars/lllyasviel/ControlNet?style=social"/> : Let us control diffusion models! "Adding Conditional Control to Text-to-Image Diffusion Models". (**[arXiv 2023](https://arxiv.org/abs/2302.05543)**).  

        - [Midjourney](https://www.midjourney.com/) : Midjourney is an independent research lab exploring new mediums of thought and expanding the imaginative powers of the human species.

        - [DreamStudio](https://dreamstudio.ai/) : Effortless image generation for creators with big dreams.

        - [Firefly](https://www.adobe.com/sensei/generative-ai/firefly.html) : Adobe Firefly: Experiment, imagine, and make an infinite range of creations with Firefly, a family of creative generative AI models coming to Adobe products.

        - [Jasper](https://www.jasper.ai/) : Meet Jasper. On-brand AI content wherever you create.

        - [Copy.ai](https://www.copy.ai/) : Whatever you want to ask, our chat has the answers.

        - [Peppertype.ai](https://www.peppercontent.io/peppertype-ai/) : Leverage the AI-powered platform to ideate, create, distribute, and measure your content and prove your content marketing ROI.

        - [ChatPPT](https://chat-ppt.com/) : ChatPPT来袭命令式一键生成PPT。








    - #### Cpp Implementation

      - [llama.cpp](https://github.com/ggerganov/llama.cpp) <img src="https://img.shields.io/github/stars/ggerganov/llama.cpp?style=social"/> : Inference of [LLaMA](https://github.com/facebookresearch/llama) model in pure C/C++.

      - [skeskinen/llama-lite](https://github.com/skeskinen/llama-lite) <img src="https://img.shields.io/github/stars/skeskinen/llama-lite?style=social"/> : Embeddings focused small version of Llama NLP model.

      - [whisper.cpp](https://github.com/ggerganov/whisper.cpp) <img src="https://img.shields.io/github/stars/ggerganov/whisper.cpp?style=social"/> : High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model.

      - [Const-me/Whisper](https://github.com/Const-me/Whisper) <img src="https://img.shields.io/github/stars/Const-me/Whisper?style=social"/> : High-performance GPGPU inference of OpenAI's Whisper automatic speech recognition (ASR) model.

      - [wangzhaode/ChatGLM-MNN](https://github.com/wangzhaode/ChatGLM-MNN) <img src="https://img.shields.io/github/stars/wangzhaode/ChatGLM-MNN?style=social"/> : Pure C++, Easy Deploy ChatGLM-6B. 

      - [ztxz16/fastllm](https://github.com/ztxz16/fastllm) <img src="https://img.shields.io/github/stars/ztxz16/fastllm?style=social"/> : 纯c++实现，无第三方依赖的大模型库，支持CUDA加速，目前支持国产大模型ChatGLM-6B，MOSS; 可以在安卓设备上流畅运行ChatGLM-6B。




    - #### Rust Implementation

      - [rustformers/llm](https://github.com/rustformers/llm) <img src="https://img.shields.io/github/stars/rustformers/llm?style=social"/> : Run inference for Large Language Models on CPU, with Rust 🦀🚀🦙.

      - [sobelio/llm-chain](https://github.com/sobelio/llm-chain) <img src="https://img.shields.io/github/stars/sobelio/llm-chain?style=social"/> : llm-chain is a collection of Rust crates designed to help you work with Large Language Models (LLMs) more effectively. [llm-chain.xyz](https://llm-chain.xyz/)

      - [coreylowman/llama-dfdx](https://github.com/coreylowman/llama-dfdx) <img src="https://img.shields.io/github/stars/coreylowman/llama-dfdx?style=social"/> : [LLaMa 7b](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) with CUDA acceleration implemented in rust. Minimal GPU memory needed! 

      - [Atome-FE/llama-node](https://github.com/Atome-FE/llama-node) <img src="https://img.shields.io/github/stars/Atome-FE/llama-node?style=social"/> : Believe in AI democratization. llama for nodejs backed by llama-rs and llama.cpp, work locally on your laptop CPU. support llama/alpaca/gpt4all/vicuna model. [www.npmjs.com/package/llama-node](https://www.npmjs.com/package/llama-node)

      - [Noeda/rllama](https://github.com/Noeda/rllama) <img src="https://img.shields.io/github/stars/Noeda/rllama?style=social"/> : Rust+OpenCL+AVX2 implementation of LLaMA inference code.

      - [tazz4843/whisper-rs](https://github.com/tazz4843/whisper-rs) <img src="https://img.shields.io/github/stars/tazz4843/whisper-rs?style=social"/> : Rust bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

      - [Cormanz/smartgpt](https://github.com/Cormanz/smartgpt) <img src="https://img.shields.io/github/stars/Cormanz/smartgpt?style=social"/> : A program that provides LLMs with the ability to complete complex tasks using plugins. 

      - [femtoGPT](https://github.com/keyvank/femtoGPT) <img src="https://img.shields.io/github/stars/keyvank/femtoGPT?style=social"/> : femtoGPT is a pure Rust implementation of a minimal Generative Pretrained Transformer. [discord.gg/wTJFaDVn45](https://github.com/keyvank/femtoGPT)

      - [pykeio/diffusers](https://github.com/pykeio/diffusers) <img src="https://img.shields.io/github/stars/pykeio/diffusers?style=social"/> : modular Rust library for optimized Stable Diffusion inference 🔮 [docs.rs/pyke-diffusers](https://docs.rs/pyke-diffusers/latest/pyke_diffusers/)





    - #### Zig Implementation
    
      - [renerocksai/gpt4all.zig](https://github.com/renerocksai/gpt4all.zig) <img src="https://img.shields.io/github/stars/renerocksai/gpt4all.zig?style=social"/> : ZIG build for a terminal-based chat client for an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa.

    



  - ### Awesome List

    - [xx025/carrot](https://github.com/xx025/carrot) <img src="https://img.shields.io/github/stars/xx025/carrot?style=social"/> : Free ChatGPT Site List. [cc.ai55.cc](https://cc.ai55.cc/)

    - [LiLittleCat/awesome-free-chatgpt](https://github.com/LiLittleCat/awesome-free-chatgpt) <img src="https://img.shields.io/github/stars/LiLittleCat/awesome-free-chatgpt?style=social"/> : 🆓免费的 ChatGPT 镜像网站列表，持续更新。List of free ChatGPT mirror sites, continuously updated. 

    - [lzwme/chatgpt-sites](https://github.com/lzwme/chatgpt-sites) <img src="https://img.shields.io/github/stars/lzwme/chatgpt-sites?style=social"/> : 搜集国内可用的 ChatGPT 在线体验免费网站列表。定时任务每日更新。[lzw.me/x/chatgpt-sites/](https://lzw.me/x/chatgpt-sites/)

    - [formulahendry/awesome-gpt](https://github.com/formulahendry/awesome-gpt) <img src="https://img.shields.io/github/stars/formulahendry/awesome-gpt?style=social"/> : A curated list of awesome projects and resources related to GPT, ChatGPT, OpenAI, LLM, and more.  

    - [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) <img src="https://img.shields.io/github/stars/Hannibal046/Awesome-LLM?style=social"/> : Awesome-LLM: a curated list of Large Language Model.

    - [cedrickchee/awesome-transformer-nlp](https://github.com/cedrickchee/awesome-transformer-nlp) <img src="https://img.shields.io/github/stars/cedrickchee/awesome-transformer-nlp?style=social"/> : A curated list of NLP resources focused on Transformer networks, attention mechanism, GPT, BERT, ChatGPT, LLMs, and transfer learning. 

    - [GT-RIPL/Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics) <img src="https://img.shields.io/github/stars/GT-RIPL/Awesome-LLM-Robotics?style=social"/> : A comprehensive list of papers using large language/multi-modal models for Robotics/RL, including papers, codes, and related websites.

    - [mikhail-bot/awesome-gpt3](https://github.com/mikhail-bot/awesome-gpt3) <img src="https://img.shields.io/github/stars/mikhail-bot/awesome-gpt3?style=social"/> :A Curated list of awesome GPT3 tools, libraries and resources.

    - [imaurer/awesome-decentralized-llm](https://github.com/imaurer/awesome-decentralized-llm) <img src="https://img.shields.io/github/stars/imaurer/awesome-decentralized-llm?style=social"/> : Repos and resources for running LLMs locally. (e.g. LLaMA, Cerebras, RWKV).

    - [csbl-br/awesome-compbio-chatgpt](https://github.com/csbl-br/awesome-compbio-chatgpt) <img src="https://img.shields.io/github/stars/csbl-br/awesome-compbio-chatgpt?style=social"/> : An awesome repository of community-curated applications of ChatGPT and other LLMs in computational biology!

    - [atfortes/LLM-Reasoning-Papers](https://github.com/atfortes/LLM-Reasoning-Papers) <img src="https://img.shields.io/github/stars/atfortes/LLM-Reasoning-Papers?style=social"/> : Collection of papers and resources on Reasoning in Large Language Models (LLMs), including Chain-of-Thought (CoT), Instruction-Tuning, and others. 

    - [yzfly/Awesome-AGI](https://github.com/yzfly/Awesome-AGI) <img src="https://img.shields.io/github/stars/yzfly/Awesome-AGI?style=social"/> : A curated list of awesome AGI frameworks, software and resources.

    - [steven2358/awesome-generative-ai](https://github.com/steven2358/awesome-generative-ai) <img src="https://img.shields.io/github/stars/steven2358/awesome-generative-ai?style=social"/> : A curated list of modern Generative Artificial Intelligence projects and services.

    - [wshzd/Awesome-AIGC](https://github.com/wshzd/Awesome-AIGC) <img src="https://img.shields.io/github/stars/wshzd/Awesome-AIGC?style=social"/> : AIGC资料汇总学习，持续更新...... 

    - [doanbactam/awesome-stable-diffusion](https://github.com/doanbactam/awesome-stable-diffusion) <img src="https://img.shields.io/github/stars/doanbactam/awesome-stable-diffusion?style=social"/> : A curated list of awesome stable diffusion resources 🌟 

    - [Yutong-Zhou-cv/Awesome-Text-to-Image](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image) <img src="https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Text-to-Image?style=social"/> : (ෆ`꒳´ෆ) A Survey on Text-to-Image Generation/Synthesis. 

    - [SeedV/generative-ai-roadmap](https://github.com/SeedV/generative-ai-roadmap) <img src="https://img.shields.io/github/stars/SeedV/generative-ai-roadmap?style=social"/> : 生成式AI的应用路线图 The roadmap of generative AI: use cases and applications.

    


  - ### Paper Overview

    - [daochenzha/data-centric-AI](https://github.com/daochenzha/data-centric-AI) <img src="https://img.shields.io/github/stars/daochenzha/data-centric-AI?style=social"/> : A curated, but incomplete, list of data-centric AI resources. "Data-centric Artificial Intelligence: A Survey". (**[arXiv 2023](https://arxiv.org/abs/2303.10158)**).

    - [KSESEU/LLMPapers](https://github.com/KSESEU/LLMPapers) <img src="https://img.shields.io/github/stars/KSESEU/LLMPapers?style=social"/> : Collection of papers and related works for Large Language Models (ChatGPT, GPT-3, Codex etc.).


  - ### Learning Resources
    - [Microsoft OpenAI](https://aka.ms/cn/LearnOpenAI): Microsoft OpenAI 学习工具包。




## Prompts
- ### 提示语（魔法）

  - [f/awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts) <img src="https://img.shields.io/github/stars/f/awesome-chatgpt-prompts?style=social"/> : This repo includes ChatGPT prompt curation to use ChatGPT better. 

  - [travistangvh/ChatGPT-Data-Science-Prompts](https://github.com/travistangvh/ChatGPT-Data-Science-Prompts) <img src="https://img.shields.io/github/stars/travistangvh/ChatGPT-Data-Science-Prompts?style=social"/> : 🚀 ChatGPT Prompts for Data Science! A repository of 60 useful data science prompts for ChatGPT.

  - [kevinamiri/Instructgpt-prompts](https://github.com/kevinamiri/Instructgpt-prompts) <img src="https://img.shields.io/github/stars/kevinamiri/Instructgpt-prompts?style=social"/> : A collection of ChatGPT and GPT-3.5 instruction-based prompts for generating and classifying text. [prompts.maila.ai/](https://prompts.maila.ai/)





## Open API

  - ### Python API

    - [gpt4free](https://github.com/xtekky/gpt4free) <img src="https://img.shields.io/github/stars/xtekky/gpt4free?style=social"/> : decentralising the Ai Industry, just some language model api's... [discord.gg/gpt4free](https://discord.gg/gpt4free)

    - [acheong08/ChatGPT](https://github.com/acheong08/ChatGPT) <img src="https://img.shields.io/github/stars/acheong08/ChatGPT?style=social"/> : Reverse Engineered ChatGPT API by OpenAI. Extensible for chatbots etc.

    - [wong2/chatgpt-google-extension](https://github.com/wong2/chatgpt-google-extension) <img src="https://img.shields.io/github/stars/wong2/chatgpt-google-extension?style=social"/> : A browser extension that enhance search engines with ChatGPT.

    - [acheong08/EdgeGPT](https://github.com/acheong08/EdgeGPT) <img src="https://img.shields.io/github/stars/acheong08/EdgeGPT?style=social"/> : Reverse engineered API of Microsoft's Bing Chat AI.


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



## Datasets

  - ### Multimodal Datasets

    - [matrix-alpha/Accountable-Textual-Visual-Chat](https://github.com/matrix-alpha/Accountable-Textual-Visual-Chat) <img src="https://img.shields.io/github/stars/matrix-alpha/Accountable-Textual-Visual-Chat?style=social"/> : "Accountable Textual-Visual Chat Learns to Reject Human Instructions in Image Re-creation". (**[arXiv 2023](https://arxiv.org/abs/2303.05983)**). [https://matrix-alpha.github.io/](https://matrix-alpha.github.io/)



## Applications


  - ### IDE
    #### 集成开发环境

    - [Cursor](https://github.com/getcursor/cursor) <img src="https://img.shields.io/github/stars/getcursor/cursor?style=social"/> : An editor made for programming with AI 🤖. Long term, our plan is to build Cursor into the world's most productive development environment. [cursor.so](https://www.cursor.so/)


  - ### Wechat
    #### 微信

    - [fuergaosi233/wechat-chatgpt](https://github.com/fuergaosi233/wechat-chatgpt) <img src="https://img.shields.io/github/stars/fuergaosi233/wechat-chatgpt?style=social"/> : Use ChatGPT On Wechat via wechaty.



  - ### Translator
    #### 翻译

    - [yetone/openai-translator](https://github.com/yetone/openai-translator) <img src="https://img.shields.io/github/stars/yetone/openai-translator?style=social"/> : The translator that does more than just translation - powered by OpenAI.



  - ### Local knowledge Base
    #### 本地知识库

    - [imClumsyPanda/langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM) <img src="https://img.shields.io/github/stars/imClumsyPanda/langchain-ChatGLM?style=social"/> : langchain-ChatGLM, local knowledge based ChatGLM with langchain ｜ 基于本地知识库的 ChatGLM 问答。基于本地知识库的 ChatGLM 等大语言模型应用实现。

    - [yanqiangmiffy/Chinese-LangChain](https://github.com/yanqiangmiffy/Chinese-LangChain) <img src="https://img.shields.io/github/stars/yanqiangmiffy/Chinese-LangChain?style=social"/> : Chinese-LangChain：中文langchain项目，基于ChatGLM-6b+langchain实现本地化知识库检索与智能答案生成。俗称：小必应，Q.Talk，强聊，QiangTalk。



  - ### Academic Field
    #### 学术领域

    - [GPTZero](https://gptzero.me/): The World's #1 AI Detector with over 1 Million Users. Detect ChatGPT, GPT3, GPT4, Bard, and other AI models.

    - [BurhanUlTayyab/GPTZero](https://github.com/BurhanUlTayyab/GPTZero) <img src="https://img.shields.io/github/stars/BurhanUlTayyab/GPTZero?style=social"/> : An open-source implementation of [GPTZero](https://gptzero.me/). GPTZero is an AI model with some mathematical formulation to determine if a particular text fed to it is written by AI or a human being.

    - [BurhanUlTayyab/DetectGPT](https://github.com/BurhanUlTayyab/DetectGPT) <img src="https://img.shields.io/github/stars/BurhanUlTayyab/DetectGPT?style=social"/> : An open-source Pytorch implementation of [DetectGPT](https://arxiv.org/pdf/2301.11305.pdf). DetectGPT is an amazing method to determine whether a piece of text is written by large language models (like ChatGPT, GPT3, GPT2, BLOOM etc). However, we couldn't find any open-source implementation of it. Therefore this is the implementation of the paper. "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature". (**[arXiv 2023](https://arxiv.org/abs/2301.11305v1)**).

    - [binary-husky/chatgpt_academic](https://github.com/binary-husky/chatgpt_academic) <img src="https://img.shields.io/github/stars/binary-husky/chatgpt_academic?style=social"/> : ChatGPT 学术优化。科研工作专用ChatGPT拓展，特别优化学术Paper润色体验，支持自定义快捷按钮，支持markdown表格显示，Tex公式双显示，代码显示功能完善，新增本地Python工程剖析功能/自我剖析功能。

    - [kaixindelele/ChatPaper](https://github.com/kaixindelele/ChatPaper) <img src="https://img.shields.io/github/stars/kaixindelele/ChatPaper?style=social"/> : Use ChatGPT to summarize the arXiv papers. 全流程加速科研，利用chatgpt进行论文总结+润色+审稿+审稿回复。 💥💥💥面向全球，服务万千科研人的ChatPaper免费网页版正式上线：[https://chatpaper.org/](https://chatpaper.org/) 💥💥💥

    - [WangRongsheng/ChatGenTitle](https://github.com/WangRongsheng/ChatGenTitle) <img src="https://img.shields.io/github/stars/WangRongsheng/ChatGenTitle?style=social"/> : 🌟 ChatGenTitle：使用百万arXiv论文信息在LLaMA模型上进行微调的论文题目生成模型。

    - [nishiwen1214/ChatReviewer](https://github.com/nishiwen1214/ChatReviewer) <img src="https://img.shields.io/github/stars/nishiwen1214/ChatReviewer?style=social"/> : ChatReviewer: use ChatGPT to review papers; ChatResponse: use ChatGPT to respond to reviewers. 💥💥💥ChatReviewer的第一版网页出来了！！！ 直接点击：[https://huggingface.co/spaces/ShiwenNi/ChatReviewer](https://huggingface.co/spaces/ShiwenNi/ChatReviewer)

    - [Shiling42/web-simulator-by-GPT4](https://github.com/Shiling42/web-simulator-by-GPT4) <img src="https://img.shields.io/github/stars/Shiling42/web-simulator-by-GPT4?style=social"/> : Online Interactive Physical Simulation Generated by GPT-4. [shilingliang.com/web-simulator-by-GPT4/](https://shilingliang.com/web-simulator-by-GPT4/)

    - [imartinez/privateGPT](https://github.com/imartinez/privateGPT) <img src="https://img.shields.io/github/stars/imartinez/privateGPT?style=social"/> :Interact privately with your documents using the power of GPT, 100% privately, no data leaks. Built with [LangChain](https://github.com/hwchase17/langchain) and [GPT4All](https://github.com/nomic-ai/gpt4all) and [LlamaCpp](https://github.com/ggerganov/llama.cpp).







  - ### Medical Field
    #### 医药领域

    - [本草[原名：华驼(HuaTuo)]](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) <img src="https://img.shields.io/github/stars/SCIR-HI/Huatuo-Llama-Med-Chinese?style=social"/> : Repo for BenTsao [original name: HuaTuo (华驼)], Llama-7B tuned with Chinese medical knowledge. 本草[原名：华驼(HuaTuo)]: 基于中文医学知识的LLaMA微调模型。本项目开源了经过中文医学指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。我们通过医学知识图谱和GPT3.5 API构建了中文医学指令数据集，并在此基础上对LLaMA进行了指令微调，提高了LLaMA在医疗领域的问答效果。 "HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge". (**[arXiv 2023](https://arxiv.org/abs/2304.06975)**). 

    - [MedSAM](https://github.com/bowang-lab/MedSAM) <img src="https://img.shields.io/github/stars/bowang-lab/MedSAM?style=social"/> : "Segment Anything in Medical Images". (**[arXiv 2023](https://arxiv.org/abs/2304.12306)**). "微信公众号「江大白」《[MedSAM在医学领域，图像分割中的落地应用（附论文及源码）](https://mp.weixin.qq.com/s/JJ0umIzJ5VKJ87A_jnDtOw)》"。

    - [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) <img src="https://img.shields.io/github/stars/microsoft/LLaVA-Med?style=social"/> : "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day". (**[arXiv 2023](https://arxiv.org/abs/2306.00890)**). "微信公众号「CVHub」《[微软发布医学多模态大模型LLaVA-Med | 基于LLaVA的医学指令微调](https://mp.weixin.qq.com/s/gzyVtbMArWDnfSzfCkxl9w)》"。



  - ### Legal Field
    #### 法律领域

    - [LaWGPT](https://github.com/pengxiao-song/LaWGPT) <img src="https://img.shields.io/github/stars/pengxiao-song/LaWGPT?style=social"/> : 🎉 Repo for LaWGPT, Chinese-Llama tuned with Chinese Legal knowledge. LaWGPT 是一系列基于中文法律知识的开源大语言模型。该系列模型在通用中文基座模型（如 Chinese-LLaMA、ChatGLM 等）的基础上扩充法律领域专有词表、大规模中文法律语料预训练，增强了大模型在法律领域的基础语义理解能力。在此基础上，构造法律领域对话问答数据集、中国司法考试数据集进行指令精调，提升了模型对法律内容的理解和执行能力。



  - ### Math Field
    #### 数学领域

    - [Progressive-Hint](https://github.com/chuanyang-Zheng/Progressive-Hint) <img src="https://img.shields.io/github/stars/chuanyang-Zheng/Progressive-Hint?style=social"/> : "Progressive-Hint Prompting Improves Reasoning in Large Language Models". (**[arXiv 2023](https://arxiv.org/abs/2304.09797)**).



  - ### Device Deployment
    #### 设备部署

    - [MLC LLM](https://github.com/mlc-ai/mlc-llm) <img src="https://img.shields.io/github/stars/mlc-ai/mlc-llm?style=social"/> : Enable everyone to develop, optimize and deploy AI models natively on everyone's devices. [mlc.ai/mlc-llm](https://mlc.ai/mlc-llm/)

    - [Lamini](https://github.com/lamini-ai/lamini) <img src="https://img.shields.io/github/stars/lamini-ai/lamini?style=social"/> : Lamini: The LLM engine for rapidly customizing models 🦙.





  - ### GUI
    #### 图形用户界面

    - [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) <img src="https://img.shields.io/github/stars/AUTOMATIC1111/stable-diffusion-webui?style=social"/> : Stable Diffusion web UI. A browser interface based on Gradio library for Stable Diffusion.

    - [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) <img src="https://img.shields.io/github/stars/oobabooga/text-generation-webui?style=social"/> : Text generation web UI. A gradio web UI for running Large Language Models like LLaMA, llama.cpp, GPT-J, Pythia, OPT, and GALACTICA.

    - [lencx/ChatGPT](https://github.com/lencx/ChatGPT) <img src="https://img.shields.io/github/stars/lencx/ChatGPT?style=social"/> : 🔮 ChatGPT Desktop Application (Mac, Windows and Linux). [NoFWL](https://app.nofwl.com/).

    - [Synaptrix/ChatGPT-Desktop](https://github.com/Synaptrix/ChatGPT-Desktop) <img src="https://img.shields.io/github/stars/Synaptrix/ChatGPT-Desktop?style=social"/> : Fuel your productivity with ChatGPT-Desktop - Blazingly fast and supercharged!

    - [Poordeveloper/chatgpt-app](https://github.com/Poordeveloper/chatgpt-app) <img src="https://img.shields.io/github/stars/Poordeveloper/chatgpt-app?style=social"/> : A ChatGPT App for all platforms. Built with Rust + Tauri + Vue + Axum.

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
    - [2023-05-08，MathGPT来了！专攻数学大模型，解题讲题两手抓](https://mp.weixin.qq.com/s/RUnJ2T9BueDnDCu91m8uPQ)
    - [2023-05-19，前哈工大教授开发的ChatALL火了！可同时提问17个聊天模型，ChatGPT/Bing/Bard/文心/讯飞都OK](https://mp.weixin.qq.com/s/1ERc9nBKMz9H_7hO02ky6w)
    - [2023-05-19，ChatGPT突然上线APP！iPhone可用、速度更快，GPT-4用量限制疑似取消](https://mp.weixin.qq.com/s/TPeViQhBPrcUqWf7LbWsNg)
    - [2023-05-28，「大一统」大模型论文爆火，4种模态任意输入输出，华人本科生5篇顶会一作，网友：近期最不可思议的论文](https://mp.weixin.qq.com/s/Mg_qnawkYSWnRHk4LIEIsQ)
  - 微信公众号「机器之心」
    - [2023-02-15，开源方案复现ChatGPT流程！1.62GB显存即可体验，单机训练提速7.73倍](https://mp.weixin.qq.com/s/j8gvD_4ViRE4WQaQlcnmrQ)
    - [2023-02-19，跟李沐学ChatGPT背后技术：67分钟读透InstructGPT论文](https://mp.weixin.qq.com/s/s5WrGn_dQyHrsZP8qsI2ag)
    - [2023-02-21，复旦发布中国版ChatGPT：MOSS开启测试冲上热搜，服务器挤爆](https://mp.weixin.qq.com/s/LjwSozikB6CK5zh2Nd2JHw)
    - [2023-03-13，清华朱军团队开源首个基于Transformer的多模态扩散大模型，文图互生、改写全拿下](https://mp.weixin.qq.com/s/B68hXlFxA9L5jiWiMrEEiA)
    - [2023-03-14，真·ChatGPT平替：无需显卡，MacBook、树莓派就能运行LLaMA](https://mp.weixin.qq.com/s/7bRwX047jkZC53KYbhKARw)
    - [2023-03-15，GPT-4震撼发布：多模态大模型，直接升级ChatGPT、必应，开放API，游戏终结了？](https://mp.weixin.qq.com/s/kA7FBZsT6SIvwIkRwFS-xw)
    - [2023-04-02，3090单卡5小时，每个人都能训练专属ChatGPT，港科大开源LMFlow](https://mp.weixin.qq.com/s/LCGQyNA6sHcdfIIARSNlww)
    - [2023-04-06，CV不存在了？Meta发布「分割一切」AI 模型，CV或迎来GPT-3时刻](https://mp.weixin.qq.com/s/-LWG3rOz60VWiwdYG3iaWQ)
    - [2023-05-14，GPT-4拿下最难数学推理数据集新SOTA，新型Prompting让大模型推理能力狂升](https://mp.weixin.qq.com/s/y8u40qIXm3oWZkvgKOV17Q)
    - [2023-05-20，有手就行？把大象P转身只需拖动鼠标，华人一作DragGAN爆火](https://mp.weixin.qq.com/s/wCvfcmv8OhGqo_fxxZUpKw)
    - [2023-05-21，北京出手通用人工智能：产业创新伙伴计划公布，要推动大模型产业加速落地](https://mp.weixin.qq.com/s/gmclRnJvFnFIc6V-zU67ng)
    - [2023-06-09，智源「悟道3.0」大模型系列问世，这次不拼参数，开源开放成为主角](https://mp.weixin.qq.com/s/kKqSa0sQOuRuQF7gDy7tIw)
    - [2023-06-10，随时随地，追踪每个像素，连遮挡都不怕的「追踪一切」视频算法来了](https://mp.weixin.qq.com/s/IqcvtfTekSKELLIjX7qRCQ)
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
    - [2023-05-14，清华大学邓志东教授——通用大模型：深度学习的极限发展](https://mp.weixin.qq.com/s/J-JMBiBDBqXmfDWwKbze5g)
    - [2023-05-21，从 GPU 到 ChatGPT](https://mp.weixin.qq.com/s/oobtNmLlvwZyheAk5jADmA)
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
    - [2023-05-12，MedSAM在医学领域，图像分割中的落地应用（附论文及源码）](https://mp.weixin.qq.com/s/JJ0umIzJ5VKJ87A_jnDtOw)
    - [2023-05-16，算法工程师如何优雅地使用ChatGPT?](https://mp.weixin.qq.com/s/FHdwnTPM6kOsMvAPcegrwg)
    - [2023-06-03，深入浅出，Stable Diffusion完整核心基础讲解](https://mp.weixin.qq.com/s/5HnOAmUKDnOtf2xDX2R9Xg)
    - [2023-06-03，分割一切模型(SAM)的全面综述调研](https://mp.weixin.qq.com/s/39imonlyIdSHYW9VnQhOjw)
    - [2023-06-10，万字长文，解析大模型在自动驾驶领域的应用](https://mp.weixin.qq.com/s/QGF8ssfB6Rk350ro-ohIHA)
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
    - [2023-05-15，Nature发文！ChatGPT加速科研编程](https://mp.weixin.qq.com/s/MoXAnTJIV4JTVppfmBccHA)
  - 微信公众号「CV技术指南」
    - [2023-04-07，3090单卡5小时，每个人都能训练专属ChatGPT，港科大开源LMFlow](https://mp.weixin.qq.com/s/h6zbAVgFpW0ccdEHjLFpdQ)
    - [2023-04-07，上线一天，4k star | Facebook：Segment Anything](https://mp.weixin.qq.com/s/G7xeuZE3vHuujQrDxIrePA)
  - 微信公众号「计算机视觉工坊」
    - [2023-04-07，超震撼！Meta发布「分割一切」AI 模型！](https://mp.weixin.qq.com/s/_IbadabLJnvv1_a-NsAJfg)
    - [2023-04-08，CV开启大模型时代！谷歌发布史上最大ViT：220亿参数，视觉感知力直逼人类](https://mp.weixin.qq.com/s/ur2WTw95pUduxh9EYULR_Q)
  - 微信公众号「新智元」
    - [2023-02-03，60天月活破亿，ChatGPT之父传奇：16岁出柜，20岁和男友一同当上CEO](https://mp.weixin.qq.com/s/W1xfLgZXWL3lfP4_54SQKw)
    - [2023-03-17，微软深夜放炸弹！GPT-4 Office全家桶发布，10亿打工人被革命](https://mp.weixin.qq.com/s/YgiurOE0uZ7lRDx1ehpbhQ)
    - [2023-05-03，AI通灵！类ChatGPT模型解码大脑信息，准确率高达82%](https://mp.weixin.qq.com/s/4KbtJ5cfur7KrWWijjQtIA)
    - [2023-05-20，GAN逆袭归来！清华校友论文引爆AI绘图圈，一秒把大象P转身，Diffusion黯然失色](https://mp.weixin.qq.com/s/DBLMAEbVw6v4xH94-5Zl3w)
  - 微信公众号「智东西」
    - [2023-02-06，ChatGPT版搜索引擎突然上线，科技巨头们坐不住了！](https://mp.weixin.qq.com/s/lncJm6hmK3AQNF2paWI5Dw)
    - [2023-04-07，ChatGPT和Matter两大风口汇合！AWE同期AIoT智能家居峰会月底举行，首批嘉宾公布](https://mp.weixin.qq.com/s/cuI8sSff_zGiLtwukAcLRw)
    - [2023-04-23，BroadLink CEO刘宗孺：ChatGPT助推全屋智能管家式变革](https://mp.weixin.qq.com/s/t4BPrvYT8oF8lGKutjpJtQ)
    - [2023-04-23，复旦MOSS升级版开源上线；马斯克启动TruthGPT；海康训练出百亿参数CV大模型丨AIGC大事周报](https://mp.weixin.qq.com/s/gBDcHw1SFSCWpJIxeC5vHg)
    - [2023-05-16，北京打响大模型地方战第一枪：公布通用人工智能发展21项措施](https://mp.weixin.qq.com/s/HdTkIaLL33ZMhrQ00fVYZQ)
  - 微信公众号「CSDN」
    - [2023-03-25，ChatGPT 已成为下一代的新操作系统！](https://mp.weixin.qq.com/s/MwrMhVydbhpP6c0AvPp8oQ)
    - [2023-04-06，CV 迎来 GPT-3 时刻，Meta 开源万物可分割 AI 模型和 1100 万张照片，1B+掩码数据集！](https://mp.weixin.qq.com/s/spBwU0UecbxbEl88SA4GJQ)
    - [2023-04-11，最爱 ChatGPT，每天编码 300 行，月薪 8k-17k 占比骤减！揭晓中国开发者真实现状](https://mp.weixin.qq.com/s/P6KjP1Xv85wSWjuxvMzK7Q)
    - [2023-05-10，在 GitHub 上“搞事”，Meta 开源 ImageBind 新模型，超越 GPT-4，对齐文本、音频等 6 种模态！](https://mp.weixin.qq.com/s/wd5vnGEQaVjpLGWYUAo-gA)
    - [2023-05-17，OpenAI CEO 在美国国会首秀：回应对 AI 的一切质疑，主动要求接受监管！](https://mp.weixin.qq.com/s/B6AXGXgwELNrG4FffTfiug)
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
    - [2023-05-19，ChatGPT的工作原理，这篇文章说清楚了](https://mp.weixin.qq.com/s/mt9RH3loOfo3--s1aKVTXg)
  - 微信公众号「新机器视觉」
    - [2023-02-13，ChatGPT 算法原理](https://mp.weixin.qq.com/s/DYRjmJ7ePTqV1RFkBZFCTw)
  - 微信公众号「投行圈子」
    - [2023-02-11，ChatGPT研究框架（80页PPT）](https://mp.weixin.qq.com/s/eGLqpTvFztok3MWE3ISc2A)
  - 微信公众号「机器学习算法那些事」
    - [2023-02-08，多模态版ChatGPT，拿下视觉语言新SOTA， 代码已开源](https://mp.weixin.qq.com/s/lsRSzwsLiTo6anPnKFa-4A)
  - 微信公众号「机器学习算法工程师」
    - [2023-04-08，CV突然进入GPT4时代！Meta和智源研究院发布「分割一切」AI 模型](https://mp.weixin.qq.com/s/9zTX0awkGPc9kfoX2QpDIg)
    - [2023-05-04，开源版Imagen来了！效果完全碾压Stable Diffusion！](https://mp.weixin.qq.com/s/Ipsw1smfINxcJT2sY00-QQ)
    - [2023-05-17，StarCoder: 最先进的代码大模型](https://mp.weixin.qq.com/s/XrY-pgBQ-DoTH_0olJ7ytw)
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
    - [2023-05-21，ChatGPT 探索：英语学习小助手](https://mp.weixin.qq.com/s/QGURRcD3QOM7-4x0CumX4Q)
    - [2023-05-25，ChatGPT 桌面应用 v1.0.0 发布啦！](https://mp.weixin.qq.com/s/jbQCws2G8hNdytIMPHHg0w)
  - 微信公众号「学术头条」
    - [2023-02-22，揭秘ChatGPT背后的AI“梦之队”：90后科研“后浪”展示强大创新能力｜智谱研究报告](https://mp.weixin.qq.com/s/sncE01utzu_-r3dLFYU5QA)
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
    - [2023-04-29，哈工大团队开源医学智能问诊大模型 | 华佗: 基于中文医学知识的LLaMa指令微调模型](https://mp.weixin.qq.com/s/YKR3Bt-Ii4M0MLJApWwyDQ)
    - [2023-06-05，X-AnyLabeling: 一款多SOTA模型集成的高精度自动标注工具！](https://mp.weixin.qq.com/s/Fi7i4kw0n_QsA7AgmtP-JQ)
    - [2023-06-07，三万字长文带你全面解读生成式AI](https://mp.weixin.qq.com/s/BDYHCnkihSChKBJHVxqywA)
    - [2023-06-08，微软发布医学多模态大模型LLaVA-Med | 基于LLaVA的医学指令微调](https://mp.weixin.qq.com/s/gzyVtbMArWDnfSzfCkxl9w)
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
  - 微信公众号「澎湃新闻」
    - [2023-05-17，莫言给余华写颁奖词，找ChatGPT帮忙](https://mp.weixin.qq.com/s/ym0w_1ftIw5BpPnGSDLsYg)
  - 微信公众号「宅码」
    - [2023-04-18，【知出乎争】GPT的变现和技术介绍](https://mp.weixin.qq.com/s/yWTriSW7CGndHraJXAi3FQ)
  - 微信公众号「Web3天空之城」
    - [2023-05-07，AI教父最新MIT万字访谈: 人类可能只是AI演化过程中的一个过渡阶段](https://mp.weixin.qq.com/s/VxlyLOUP_CIyMvGCBGimCQ)
    - [2023-05-17，Sam Altman 国会质询2.5万字全文：如果这项技术出错，它会出错得很严重](https://mp.weixin.qq.com/s/DqPTN8pADPWGjMSiO3__2w)
  - 微信公众号「AI前线」
    - [2023-05-03，7天花5万美元，我们成功复制了 Stable Diffusion，成本大降88%！训练代码已开源](https://mp.weixin.qq.com/s/KYhjUOhi3dBvGptBiBlW8A)
  - 微信公众号「CVer」
    - [2023-05-03，代季峰教授：超大规模视觉通用模型最新研究成果分享](https://mp.weixin.qq.com/s/RYCHY0CrFbnM88ORegED1A)
    - [2023-05-20，华人一作DragGAN爆火！拖动你的GAN：交互式图像编辑新高度](https://mp.weixin.qq.com/s/QGyuCPFzg2W2QUyMu4HD2g)
  - 微信公众号「Jack Cui」
    - [2023-05-04，新项目又火了，已开源！gpt4免费了...](https://mp.weixin.qq.com/s/f6Sxc1ZYWguYkiFV3atI3g)
    - [2023-05-16，一个厉害的中医GPT，AI老中医开源了！](https://mp.weixin.qq.com/s/9O1pr7UZVRz9G9D8kMvwRw)
    - [2023-05-19，狂飙，ChatGPT 官方 iOS 应用上线了！](https://mp.weixin.qq.com/s/dt3Rf7j7ALt-GxnAXxnOgQ)
  - 微信公众号「AI数据派」
    - [2023-05-05，UC伯克利发布大语言模型排行榜！Vicuna夺冠，清华ChatGLM进前5](https://mp.weixin.qq.com/s/JS2ISYUOiSQKECYuXB8h5A)
  - 微信公众号「我爱计算机视觉」
    - [2023-05-05，图文理解能力强大！多模态对话生成模型：mPLUG-Owl，已开源！](https://mp.weixin.qq.com/s/tQYV54g6aMJxogmI3MzmiA)
  - 微信公众号「计算机视觉联盟」
    - [2023-05-10，北大、西湖大学等开源PandaLM](https://mp.weixin.qq.com/s/mKq56QrTWTd7IiXcmYqSFA)
  - 微信公众号「机器学习与AI生成创作」
    - [2023-05-09，借助通用分割大模型！半自动化标注神器，Label-Studio X SAM（附源码）](https://mp.weixin.qq.com/s/2qPiEkuruIVZk1HcTqHYjg)
  - 微信公众号「差评」
    - [2023-04-17，我有个周入百万的项目：教人用ChatGPT。](https://mp.weixin.qq.com/s/awfe5Hb2_g-EZ-rHJY-SBw)
  - 微信公众号「程序员的那些事」
    - [2023-05-16，Midjourney 5.1 震撼更新！逼真到给跪，中国情侣细节惊艳，3D视频大片马上来](https://mp.weixin.qq.com/s/IViZPmfKlzgc83ozuj-zcg)
  - 微信公众号「51CTO技术栈」
    - [2023-05-19，Stability AI开源一系列人工智能应用](https://mp.weixin.qq.com/s/QOT7ycS5MuobPW2XeYWLWw)
    - [2023-05-16，入驻QQ一天就爆满！Midjourney中文版来了！](https://mp.weixin.qq.com/s/2eLc_vIUIdR9wKIUzOxZ0A)
  - 微信公众号「GitHubDaily」
    - [2023-05-18，人手一个 Midjourney，StableStudio 重磅开源！](https://mp.weixin.qq.com/s/SbW3drfTmXyoeuwpDg5o2w)
  - 微信公众号「CreateAMind」
    - [2023-05-20，改进GPT的底层技术](https://mp.weixin.qq.com/s/5zZrol7CLHD-kEMejwHimw)
  - 微信公众号「深度学习与NLP」
    - [2023-05-21，邱锡鹏团队提出具有跨模态能力SpeechGPT，为多模态LLM指明方向](https://mp.weixin.qq.com/s/fEBWELAiEJikC91pwk9l-Q)
  - 微信公众号「APPSO」
    - [2023-06-01，ChatGPT路线图曝光：没有GPT-5、识图功能要等到明年、GPT-3或将开源](https://mp.weixin.qq.com/s/yKst4w3x0II3kGy5VqY2gA)
  - 微信公众号「夕小瑶科技说」
    - [2023-05-31，一个技巧，让ChatGPT学会复杂编程，编程水平逼近人类程序员！](https://mp.weixin.qq.com/s/QgL5-fTA99InHsoI7hJ8lw)
  - 微信公众号「佐思汽车研究」
    - [2023-05-26，大模型上不了车](https://mp.weixin.qq.com/s/guxGFY5Jg_YdWDxnIyTZsA)   
  - 微信公众号「开源技术服务中心」
    - [2023-05-31，河套IT WALK(总第64期)：AI与自动驾驶科技：打造未来生活方式](https://mp.weixin.qq.com/s/wGupibJ9cKrjdSbUv9cQgQ)   
  - 微信公众号「OneFlow」
    - [2023-06-09，GPT总设计师：大型语言模型的未来](https://mp.weixin.qq.com/s/DAV4ZQ5HVKw3z-mQnM7cWA)   



