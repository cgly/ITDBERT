

import torch

print(torch.cuda.is_available())

############################################################
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers import BertWordPieceTokenizer

#paths = [str(x) for x in Path("NLPCorpus").glob("**/*.txt")]
paths=r"E:\code\zhuheCode\act-Classification\wasa_data\330CertAdd.txt"
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, vocab_size=600, min_frequency=1, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model(r"E:\code\zhuheCode\act-Classification\RebertaMlm\42_H1_24_42")
############################################################
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=600,
    max_position_embeddings=512,
    num_attention_heads=8,
    num_hidden_layers=1,
    type_vocab_size=1,
)
############################################################
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(r"E:\code\zhuheCode\act-Classification\RebertaMlm\42_H1_24_42", max_len=512)
############################################################
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model.num_parameters())
############################################################
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=r"E:\code\zhuheCode\act-Classification\wasa_data\330CertAdd.txt",
    block_size=128,
)
############################################################
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
############################################################
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=r"E:\code\zhuheCode\act-Classification\RebertaMlm\42_H6_24_42",
    overwrite_output_dir=True,
    num_train_epochs=200,
    per_gpu_train_batch_size=128,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()

trainer.save_model(r"E:\code\zhuheCode\act-Classification\RebertaMlm\42_H1_24_42")