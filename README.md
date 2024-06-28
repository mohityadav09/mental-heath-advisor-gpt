
# Mental Health Advisor GPT

This project involves the fine-tuning of the `[Mistral-7B-Instruct-v0.2-GPTQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ)` large language model using Quantized Low-Rank Adaptation (QLoRA) techniques on the `mental_health_counseling_conversations conversations` dataset. The goal is to enhance the model's ability to generate contextually appropriate and empathetic responses in mental health counseling scenarios.

## Project Overview

- **Model**: Mistral-7B-Instruct-v0.2-GPTQ
- **Technique**: QLoRA
- **Dataset**: mental_health_counseling_conversations
- **Objective**: To create a robust mental health advisory system capable of providing empathetic and contextually relevant responses.

## Key Features

- Comprehensive data preprocessing including tokenization, incorporation of system prompts, and data splitting into training and validation sets.
- Implementation of QLoRA to optimize computational efficiency and model performance.
- Ongoing efforts to optimize hyperparameters and explore advanced evaluation metrics to further enhance model accuracy and effectiveness.



## Installation

#### Installation of required Libraries


```bash
  pip install transformers -q
```
```bash
  pip install peft
```
```bash
  pip install bitsandbytes
```
```bash
  pip install optimum
```
```bash
  pip install auto-gptq
```
```bash
  pip install torch
```

    
## Usage/Examples
The fine-tuned model is available on Hugging Face: [mohityadav/mental-health-advisorGpt](https://huggingface.co/mohityadav/mental-health-advisorGpt).

To use the model, you can load it directly from Hugging Face:


```python
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
```
```python

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("mohityadav/mental-health-advisorGpt")
model = PeftModel.from_pretrained(model, "mohityadav/mental-health-advisorGpt")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

```

```python
instructions_string = """
You are a compassionate and knowledgeable mental health advisor. Someone is sharing their mental 
state or expressing doubts about their well-being. Provide the best advice and suggestions to help them navigate their
feelings. Offer empathetic, practical, and actionable guidance, and recommend helpful resources or techniques.
Ensure your response is simple, relatable, encourages openness, and provides reassurance.
Please provide valuable advice for this comment.
"""
comment="i think i am mentally sick and weird"

prompt=f"[INST] {instructions_string} \n{comment} \n[/INST]"

inputs=tokenizer(prompt,return_tensors='pt')

output=model.generate(input_ids=inputs["input_ids"].to('cuda'), max_new_tokens=140)

print(tokenizer.batch_decode(output)[0])
```

## Training hyperparameters

| Hyperparameter                  | Value                          |
|---------------------------------|--------------------------------|
| learning_rate                   | 0.0002                         |
| train_batch_size                | 4                              |
| eval_batch_size                 | 4                              |
| seed                            | 42                             |
| gradient_accumulation_steps     | 4                              |
| total_train_batch_size          | 16                             |
| optimizer                       | Adam                           |
| optimizer_betas                 | (0.9, 0.999)                   |
| optimizer_epsilon               | 1e-08                          |
| lr_scheduler_type               | linear                         |
| lr_scheduler_warmup_steps       | 2                              |
| num_epochs                      | 10                             |
| mixed_precision_training        | Native AMP                     |
