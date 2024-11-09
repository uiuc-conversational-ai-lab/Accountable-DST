import json
import os
import argparse
import time
from tqdm import tqdm
import logging
import time
import argparse
import transformers
import torch
import torch
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import get_prompt, get_slot_list, get_mwz_test_dataset, get_mwz_dev_dataset
from utils import get_snips_details, get_prompt_snips, get_snips_infer_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.functional import sigmoid
os.environ["WANDB_DISABLED"] = "true"

#------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-model','--model', help='Model name', default='meta-llama/Meta-Llama-3.1-8B-Instruct', required=False)
parser.add_argument('-best','--best', help='Path of best pre-trained model checkpoint.', required=True)
parser.add_argument('-slot','--slot', help='Use slot classification', default=False, action='store_true')
parser.add_argument('-check','--check', help='Test run on few samples', default=False, action='store_true')
parser.add_argument('-m','--m', help='Weight of classification loss', type=float, required=False, default=0.25)
parser.add_argument('-data','--data', help='Dataset (mwz/snips)', type=str, default='mwz', required=False, choices=['mwz', 'snips'])
parser.add_argument('-dev','--dev', help='Generate output for dev set instead of test set', default=False, action='store_true')

args = vars(parser.parse_args())
model_dir = args['path']
model_name = args['model']
best_model_ckpt =  args['best']
use_classification_head = args['slot']
test_run = args['check']
c_wt = args['m']
dataset_name = args['data']
is_dev = args['dev']

if "llama" in model_name.lower():
    model_type = "llama"
elif "mistral" in model_name.lower():
    model_type = "mistral"
elif "gemma" in  model_name.lower():
    model_type = "gemma"
else:
    model_type = "other"
    
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
else:
    print(f"Model directory {model_dir} do no exist.")
    exit(0)
    
#Setting log file
if(is_dev):
    log_file = os.path.join(model_dir, 'log_gen_dev.txt')
else:
    log_file = os.path.join(model_dir, 'log_gen_test.txt')
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

# Check the path of best model checkpoint
if(not os.path.isdir(best_model_ckpt)):
    print(f"Model checkpoint does not exist. Exiting the code.")
    exit(0)

model = DstModel.from_pretrained(best_model_ckpt)
model.to(device)
model.eval()
logger.info("Best model loaded")
logger.info("-"*30)

if(dataset_name=="snips"):
    test_dataset = get_snips_infer_dataset(test_run, conv_limit, logger, slot_list, dom_slot_dict, is_dev)
else:
    if(is_dev):
        test_dataset = get_mwz_dev_dataset(test_run, conv_limit, logger,  model_type, tokenizer)
    else:
        test_dataset = get_mwz_test_dataset(test_run, conv_limit, logger,  model_type, tokenizer)
logger.info(test_dataset)
logger.info("Test data loaded")
logger.info("-"*30)

# Inference
lst_result = []
print("Testing ...")
logger.info("Testing Started...")
for dt in tqdm(test_dataset["test"]):
    conv_history = dt["utt"]
    bs = dt["bs"]
    slot_ids = dt["slots"]

    if(dataset_name=="snips"):
        prompt = get_prompt_snips(conv_history, domains, slot_details, model_type, tokenizer)
    else:
        prompt = get_prompt(conv_history, model_type, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    idx = input_ids.size(1)

    with torch.no_grad():
        output = model.lm.generate(input_ids=input_ids, max_new_tokens=256, output_scores=True, output_logits=True, return_dict_in_generate=True)
        resp_idx = output.sequences[0][idx:]
        out_scores = output.scores
        logit_scores = output.logits

        lst_scores = []
        lst_logits = []
        for i in range(len(resp_idx)):
            lst_scores.append(round(out_scores[i][0][resp_idx[i]].item(),4))
            lst_logits.append(round(logit_scores[i][0][resp_idx[i]].item(),4))    

        bs_pred = tokenizer.decode(resp_idx, skip_special_tokens=True)
        bs_gt = json.loads(bs)

        if(use_classification_head):
            lm_out = model.lm(input_ids=input_ids, output_hidden_states=True)
            hidden = lm_out.hidden_states[-1]

            h = hidden[0][idx-1].float()
            c_logits = model.c_head(h)
            slot_prob = sigmoid(c_logits).cpu().detach().tolist()
            slot_prob = [round(v,4) for v in slot_prob]

            slot_prob_dict = {}
            for j in range(len(slot_prob)):
                slot_prob_dict[slot_list[j]] = slot_prob[j]
        
        res_dict = {}
        res_dict["utt"] = conv_history
        res_dict["bs_gt"] = bs_gt
        res_dict["bs_pred"] = bs_pred
        res_dict["resp_idx"] = resp_idx.tolist()
        res_dict["scores"] = lst_scores
        res_dict["logits"] = lst_logits
        if(use_classification_head):
            res_dict["prob"] = slot_prob_dict
        lst_result.append(res_dict)

out_filname = "result_output_dev.json" if is_dev else "result_output.json"
with open(os.path.join(model_dir, out_filname), 'w', encoding='utf-8') as f:
    json.dump(lst_result, f, ensure_ascii=False, indent=4)

logger.info("Testing Complete")
print("done")
#------------------------------------------------
