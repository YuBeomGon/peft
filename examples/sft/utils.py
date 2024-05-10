import os
import json
from enum import Enum

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig


DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
EndofSeq = "<|im_end|>"

class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]
    
def create_datasets_from_db(tokenizer, data_args, training_args):
    def make_align_prompt(example):
        content = ""
        if 'system' in example:
            content += f"""\n\nSystem:{example["system"]} {EndofSeq}"""
        if 'human' in example:
            content += f"""\n\nHuman:{example["human"]} {EndofSeq}"""        
        if 'assistant' in example:
            content += f"""\n\nAssistant:\n {example["assistant"]} {EndofSeq}"""           
        example['content'] = content
        return example
    
    def make_chat_prompt(example):
        content = ""
        # print(type(example['content']))
        # print(example['content'])
        # example = json.loads(example['content'])
        for e in example['content']:
            if e['from'] in ('system','System'):
                content += f"\n\nSystem: {e['value']} {EndofSeq}"
            if e['from'] in ('user', 'human'):
                content += f"\n\nHuman: {e['value']} {EndofSeq}"
            if e['from'] in ('gpt', 'assistant', 'bot'):
                content += f"\n\nAssistant: {e['value']} {EndofSeq}"
                 
        example['content'] = content
        return example    

    db_url = f"{data_args.database_url}"
    table_names = f"{data_args.database_table_name}"
    table_names = table_names.split(',')

    ds_list = []
    for table in table_names:
        if table == "alignment_table":
            tmp = Dataset.from_sql(f'select system, human, assistant, src from {table};', db_url)
        elif table == "chat_table":
            tmp = Dataset.from_sql(f'select content, src from {table};', db_url)
        ds_list.append(tmp)
        
    align_ds = []
    chat_ds = []
    
    for table, ds in zip(table_names, ds_list):
        if 'alignment' in table:
            align_ds.append(ds)
        elif 'chat' in table:
            chat_ds.append(ds)    
            
    align_ds = concatenate_datasets(align_ds)
    chat_ds = concatenate_datasets(chat_ds)       
    
    if align_ds:
        align_ds = concatenate_datasets(align_ds)
        raw_align_dataset = DatasetDict({'train':align_ds})
        align_remove_columns = set(raw_align_dataset['train'].column_names) - set(['content', 'src'])
        raw_align_dataset = raw_align_dataset.map(make_align_prompt, remove_columns=align_remove_columns)
        raw_align_dataset = raw_align_dataset.shuffle()
        print('raw_align_dataset', raw_align_dataset)
        print(raw_align_dataset['train'][:2])
    if chat_ds:
        chat_ds = concatenate_datasets(chat_ds)
        raw_chat_dataset = DatasetDict({'train':chat_ds})
        chat_remove_columns = set(raw_chat_dataset['train'].column_names) - set(['content', 'src'])
        raw_chat_dataset = raw_chat_dataset.map(make_chat_prompt, remove_columns=chat_remove_columns)
        raw_chat_dataset = raw_chat_dataset.shuffle()
        print('raw_chat_dataset', raw_chat_dataset)
        print(raw_chat_dataset['train'][:2])
    if align_ds and chat_ds:
        raw_datasets = concatenate_datasets([raw_align_dataset['train'], raw_chat_dataset['train']])
        raw_datasets = DatasetDict({'train':raw_datasets})
        raw_datasets = raw_datasets.shuffle()
    elif align_ds:
        raw_datasets = raw_align_dataset
    elif chat_ds:
        raw_datasets = raw_chat_dataset
    else:
        print("*****warning one dataset should be selected")
    
    print('raw_datasets', raw_datasets)
    print(raw_datasets['train'][:2])
    
    raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1)
    

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[:2]}")
    print(f"A sample of train dataset: {valid_data[:2]}")

    return train_data, valid_data


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer
