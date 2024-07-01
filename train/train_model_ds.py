# %%
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "True"

# %%
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/no_robots", trust_remote_code=True)
ds

# %%
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


model_id = "openai-community/gpt2-large"



# %%
tokenizer = AutoTokenizer.from_pretrained(model_id, use_deepspeed=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)


# %%
# 确保填充标记设置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def preprocess(examples):
    # 从 messages 中提取文本内容进行拼接
    def extract_text(messages):
        return " ".join([msg["content"] for msg in messages])

    inputs = ["summarize: " + prompt + " " + extract_text(messages) for prompt, messages in zip(examples["prompt"], examples["messages"])]
    # 对输入和标签进行相同的填充和截断
    model_inputs = tokenizer(inputs, max_length=256, padding='max_length', truncation=True)
    labels = tokenizer(text_target=examples["category"], max_length=256, padding='max_length', truncation=True)

    # 设置 input_ids 和 labels，确保它们长度一致
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# 处理数据集
tokenized_dataset = ds.map(preprocess, batched=True, remove_columns=["prompt", "prompt_id", "messages", "category"])




training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to=None,
    fp16=True,
)


# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)
trainer.train()


