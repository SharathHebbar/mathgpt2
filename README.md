# MathGPT2

This model is a finetuned version of ```gpt2``` using ```ArtifactAI/arxiv-math-instruct-50k```

## Technical Details
1. The model's repository is in HuggingFace Link: https://huggingface.co/Sharathhebbar24/math_gpt2/tree/main

2. The model was trained on the following Specs
- Nvidia RTX 3050 4GB VRAM
- 16GB RAM
- Intel i5 12th gen 12500H

3. It took around 4:45:20 to train the model for 3 epochs with a loss of 1.63.

4. It was trained on FP16 bit architecture with a batch size of 4.

| license | datasets | language |
| ------- | -------- | -------- |
| apache-2.0 | ArtifactAI/arxiv-math-instruct-50k | en |

5. You can track the models graph here [Graphs](https://api.wandb.ai/links/sharathhebbar/j1dyilpd)

## Model description

GPT-2 is a transformers model pre-trained on a very large corpus of English data in a self-supervised fashion. This
means it was pre-trained on the raw texts only, with no humans labeling them in any way (which is why it can use lots
of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely,
it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence,
shifting one token (word or piece of word) to the right. The model uses a masking mechanism to make sure the
predictions for the token `i` only use the inputs from `1` to `i` but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks. The model is best at what it was trained for, however, which is generating texts from a
prompt.

### To use this model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Sharathhebbar24/math_gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def generate_text(prompt):
 inputs = tokenizer.encode(prompt, return_tensors='pt')
 outputs = mod1.generate(inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)
 generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
 return generated[:generated.rfind(".")+1]
prompt = "What structure is classified as a definite lie algebra?"
res = generate_text(prompt)
res
```