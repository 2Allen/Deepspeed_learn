# %%
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# 40系顯卡必須使用
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ['TOKENIZERS_PARALLELISM'] = "True"

# 載入資料集
ds = load_dataset("HuggingFaceH4/no_robots", trust_remote_code=True)
ds


# 載入模型
model_id = "openai-community/gpt2-large"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_deepspeed=True)


# 語言模型標記
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# 資料集處理
def preprocess(examples):
    def extract_text(messages):
        return " ".join([msg["content"] for msg in messages])

    inputs = ["summarize: " + prompt + " " + extract_text(messages) for prompt, messages in zip(examples["prompt"], examples["messages"])]
    model_inputs = tokenizer(inputs, max_length=256, padding='max_length', truncation=True)
    labels = tokenizer(text_target=examples["category"], max_length=256, padding='max_length', truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_dataset = ds.map(preprocess, batched=True, remove_columns=["prompt", "prompt_id", "messages", "category"])


# 設定train參數
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


# 訓練模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)
trainer.train()


