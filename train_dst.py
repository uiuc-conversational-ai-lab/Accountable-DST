import json
import os
import pandas as pd
import argparse
import time
import logging
import time
import transformers
import torch
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import Trainer, default_data_collator, EarlyStoppingCallback
from utils import get_mwz_dataset, get_prompt, get_slot_list
from utils import get_snips_details, get_prompt_snips, get_snips_dataset
from peft import LoraConfig, get_peft_model
os.environ["WANDB_DISABLED"] = "true"

#------------------------------------------------

#Model Names
## LLama: meta-llama/Meta-Llama-3.1-8B-Instruct
## Mistral: mistralai/Mistral-7B-Instruct-v0.1
## Gemma: google/gemma-7b-it

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-model','--model', help='Model name', default='meta-llama/Meta-Llama-3.1-8B-Instruct', required=False)
parser.add_argument('-slot','--slot', help='Use slot classification', default=False, action='store_true')
parser.add_argument('-check','--check', help='Test run on few samples', default=False, action='store_true')
parser.add_argument('-epochs','--epochs', help='Number of epochs', type=int, required=False, default=4)
parser.add_argument('-batch','--batch', help='Per device batch size', type=int, required=False, default=1)
parser.add_argument('-grad_acc','--grad_acc', help='Gradient Accumulation', type=int, required=False, default=4)
parser.add_argument('-lr','--lr', help='Learning rate', type=float, required=False, default=5e-5)
parser.add_argument('-m','--m', help='Weight of classification loss', type=float, required=False, default=0.25)
parser.add_argument('-train_limit','--train_limit', help='No. of samples if not training with all data. Default is 0 that indicates all data.', type=int, required=False, default=0)
parser.add_argument('-data','--data', help='Dataset (mwz/snips)', type=str, default='mwz', required=False, choices=['mwz', 'snips'])

args = vars(parser.parse_args())
model_dir = args['path']
model_name = args['model']
use_classification_head = args['slot']
test_run = args['check']
num_train_epochs = args['epochs']
batch_size = args['batch']
grad_acc = args['grad_acc']
learning_rate = args['lr']
c_wt = args['m']
train_limit = args['train_limit']
dataset_name = args['data']
train_all = True if(train_limit==0) else False

if "llama" in model_name.lower():
    model_type = "llama"
elif "mistral" in model_name.lower():
    model_type = "mistral"
elif "gemma" in  model_name.lower():
    model_type = "gemma"
else:
    model_type = "other"
    print("Your model name is not Llama/Gemma/Mistral. Set model-specific prompt and hidden size.")

lst_label_names = ["labels"]
if(use_classification_head):
    lst_label_names.append("slot_ids")
    lst_label_names.append("idx")

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print(f"Using GPU!")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
if(os.path.isdir(model_dir)):
    print(f"Model Directory {model_dir} exists.")
    exit(0)
else:
    os.mkdir(model_dir)
    print(f"Model directory {model_dir} created.")
    
    
#Setting log file
log_file = os.path.join(model_dir, 'log.txt')
logging.basicConfig(filename=log_file, filemode='a', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity(logging.INFO)
#transformers.utils.logging.set_verbosity(logging.WARNING) 
log_dir = os.path.join(model_dir, "logs")

logger.info(args)

if torch.cuda.is_available():
    logger.info(f"Using GPU!")
else:
    logger.info('No GPU available, using the CPU instead.')

logger.info("-"*30)

#------------------------------------------------

# Model settings

config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

if(dataset_name=="snips"):
    domains, slot_list, slot_details, dom_slot_dict = get_snips_details()
    conv_limit = 32
else:
    slot_list = get_slot_list()
    conv_limit = 5
    
hidden_size = 4096
if(model_type=="gemma"):
    hidden_size = 3072
num_class = len(slot_list)

logger.info(f"dataset: {dataset_name}")
logger.info(f"model_type: {model_type}")
logger.info(f"slot_list: {slot_list}")
logger.info(f"test_run: {test_run}")
logger.info(f"hidden_size: {hidden_size}")
logger.info(f"num_class: {num_class}")
logger.info(f"lst_label_names: {lst_label_names}")
logger.info("-"*30)

#------------------------------------------------

# Model Class

class DstConfig(PretrainedConfig):
    def __init__(self, 
                 model_name = model_name, 
                 use_classification_head: bool = use_classification_head, 
                 hidden_size: int = hidden_size,
                 num_class: int = num_class,
                 **kwargs,
        ):
        self.model_name = model_name
        self.use_classification_head = use_classification_head
        self.hidden_size = hidden_size
        self.num_class = num_class
        super().__init__(**kwargs)
        
class DstModel(PreTrainedModel):
    config_class = DstConfig
    
    def __init__(self, config):
        super().__init__(config)
        base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config = bnb_config, token = HF_TOKEN)
        self.lm = get_peft_model(base_model, peft_config)
        self.use_classification_head = config.use_classification_head
        self.hidden_size = config.hidden_size
        self.num_class = config.num_class
        if(config.use_classification_head):
            self.c_head = torch.nn.Linear(config.hidden_size, config.num_class)
            self.loss_bce = torch.nn.BCEWithLogitsLoss()
            
        
    def forward(self, input_ids, labels, slot_ids = None, idx = None):
        if(self.use_classification_head):
            # LM loss
            lm_out = self.lm(input_ids=input_ids, labels=labels, output_hidden_states=True)
            lm_loss = lm_out.loss
            
            # Classification head
            hidden = lm_out.hidden_states[-1] #(batch, hidden_size)
            
            c_loss = 0
            for i in range(idx.size(0)): #batch
                c_hidden = hidden[i][idx[i].item()].float().to(slot_ids.device)
                c_logits = self.c_head(c_hidden) #(num_class)
                s_loss = self.loss_bce(c_logits, slot_ids[i].float())
                c_loss += s_loss
            c_loss = (c_loss/idx.size(0)).to(lm_loss.device)
   
            # Final loss
            loss = c_wt*c_loss + lm_loss
            return {'loss': loss, 'lm_loss': lm_loss, 'c_loss': c_loss}
        else:
            lm_out = self.lm(input_ids=input_ids, labels=labels)
            loss = lm_out.loss
            return {'loss': loss}
    
#------------------------------------------------

# Functions

def model_init():
    dst_config = DstConfig(model_name, use_classification_head, hidden_size, num_class)
    model = DstModel(dst_config)
    return model

def my_data_collator(batch):
    list_len = [len(dt["input_ids"]) for dt in batch]
    n = max(list_len)
    for i in range(len(list_len)):
        batch[i]["input_ids"].extend([tokenizer.pad_token_id]*(n-list_len[i]))
        batch[i]["labels"].extend([-100]*(n-list_len[i]))
    return default_data_collator(batch)

def prepare_features(dt):
    conv_history = dt["utt"]
    belief_states = dt["bs"]
    
    if(dataset_name=="snips"):
        prompt = get_prompt_snips(conv_history, domains, slot_details, model_type, tokenizer)
    else:
        prompt = get_prompt(conv_history, model_type, tokenizer)
    
    input_ids = tokenizer.encode(prompt)
    resp_ids = tokenizer.encode(belief_states)
    resp_ids.append(tokenizer.eos_token_id)

    n = len(input_ids)
    labels = [-100]*n
    labels.extend(resp_ids[1:])
    input_ids.extend(resp_ids[1:])
    
    output = {}
    output["input_ids"] = input_ids
    output["labels"] = labels
    
    if(use_classification_head):
        output["slot_ids"] = dt["slots"]
        output["idx"] = n-1
    return output

#------------------------------------------------

# Load training data 
logger.info("Loading Data ...")
if(dataset_name=="snips"):
    dataset = get_snips_dataset(test_run, conv_limit, logger, slot_list, dom_slot_dict)
else:
    dataset = get_mwz_dataset(test_run, conv_limit, logger, model_type, tokenizer, train_all, train_limit)
logger.info(dataset)

tokenized_datasets = dataset.map(prepare_features, remove_columns=dataset["train"].column_names)
logger.info("Tokenized dataset ...")
logger.info(tokenized_datasets)
logger.info("Data Loaded ...")
logger.info("-"*30)

# Training data snippet 
logger.info("Tokenized Data Snippet:-")
idx=1
logger.info(f"input_ids: {tokenized_datasets['train']['input_ids'][idx]}")
logger.info(f"input_ids: {tokenizer.decode(tokenized_datasets['train']['input_ids'][idx], skip_special_tokens=False)}")
lbl = tokenized_datasets["train"]["labels"][idx]
lbl = [v if v!=-100 else tokenizer.eos_token_id for v in lbl]
logger.info(f"labels: {tokenizer.decode(lbl, skip_special_tokens=True)}")

if(use_classification_head):
    logger.info(f"slots: {tokenized_datasets['train']['slot_ids'][idx]}")
    i=0
    for v in tokenized_datasets['train']['slot_ids'][idx]:
        if(v==1):
            logger.info(slot_list[i])
        i+=1

logger.info("-"*30)
logger.info("-"*30)

# Training
training_args = TrainingArguments(
    output_dir=model_dir,
    seed=0,
    label_names = lst_label_names,
    prediction_loss_only = True,
    evaluation_strategy="epoch",
    logging_dir = log_dir,
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc,
    save_total_limit=1,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    report_to=None
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=my_data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 1)]
)
logger.info("Training...")
trainer.train()
logger.info("Training Done")

df_log = pd.DataFrame(trainer.state.log_history)
fl = os.path.join(model_dir, "log_history.csv")
df_log.to_csv(fl)  

best_model_ckpt = trainer.state.best_model_checkpoint
logger.info(f"Best model: {best_model_ckpt}")

logger.info("Training Complete")
logger.info("-"*30)

print("done")
#------------------------------------------------
