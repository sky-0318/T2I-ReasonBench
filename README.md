# T2I-ReasonBench

## :mega: Overview
Text-to-image (T2I) generative models have achieved remarkable progress, demonstrating exceptional capability in synthesizing high-quality images from textual prompts. While existing research and benchmarks has extensively evaluated the ability of T2I models to follow the literal meaning of prompts, spanning from knowledge coverage to concept composition, and from concise to dense prompts, their ability for reasoning over prompts to uncover implicit meaning and contextual nuances remains underexplored. To bridge this gap, we introduce ReasonBench, a novel benchmark designed to explore the reasoning boundaries of T2I models.
ReasonBench comprises 800 meticulously crafted prompts organized into four dimensions: (1) Idiom interpretation, (2) Textual image design, (3) Entity-reasoning, and (4) Scientific-reasoning. These dimensions challenge models to infer latent meaning, integrate domain knowledge, and resolve contextual ambiguities. To quantify the performance, we propose an evaluation framework leveraging a multimodal large language model (MLLM) to assess both semantic alignment and aesthetic quality of the generated images. Experiments across 13 state-of-the-art T2I models reveal that open-source models exhibit critical limitations in reasoning-driven generation, while proprietary models like GPT-Image-1 demonstrate stronger reasoning and knowledge integration. Our findings underscore the necessity of improving reasoning capabilities and structured knowledge incorporation in next-generation T2I systems. This work provides a foundational benchmark and evaluation protocol to guide future research toward reasoning-informed text to image synthesis.


## :blue_book: Prompt Suite
ReasonBench comprises 800 meticulously crafted prompts organized into four dimensions: (1) Idiom interpretation, (2) Textual image design, (3) Entity-reasoning, and (4) Scientific-reasoning, each with 200 prompts. 

Text prompts of each category are saved in a text file in the ```prompts/``` directory.


## :speech_balloon: MLLM-based Evaluation
We use **Qwen2.5-vl** as the MLLM model to evaluate the four dimensions.
### :hammer: 1. Install Requirements

MLLM-based evaluation metrics are based on the official repository of Qwen2.5-vl. You can refer to [Qwen2.5-vl's GitHub repository](https://github.com/QwenLM/Qwen2.5-VL) for specific environment dependencies and weights.

### :clapper: 2. Prepare Evaluation Images

Generate images of your model using the T2I-ReasonBench prompts provided in the `prompts` directory. Organize them in the following structure (using the *idiom interpretation* category as an example):

```
../video/idiom_interpretation
├── 0001.png
├── 0002.png
├── 0003.png
├── 0004.png
...
└── 0200.png
```

### :running: 3. Run the Evaluation Codes

After obtaining the official Qwen2.5-vl code, place the following evaluation scripts in the `Qwen2.5-VL/` directory:

- `eval_idiom.py`
- `eval_textual_image.py`
- `eval_entity.py`
- `eval_scientific.py`

Replace the image folder path and prompt folder path in the script. The evaluation code will generate a csv file with scores for each generated image, average to get the accuracy score and aesthetic score for the model.
