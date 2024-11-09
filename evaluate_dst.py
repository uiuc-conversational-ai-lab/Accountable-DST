import json
import os
import argparse
from fuzzywuzzy import fuzz
from utils import get_slot_list, get_prompt
from utils import get_snips_details, get_prompt_snips, get_snips_dataset
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-model','--model', help='Model name', default='meta-llama/Meta-Llama-3.1-8B-Instruct', required=False)
parser.add_argument('-best','--best', help='Path of best pre-trained model checkpoint.', required=True)
parser.add_argument('-m','--m', help='Weight of classification loss', type=float, required=False, default=0.25)
parser.add_argument('-slot','--slot', help='Use slot classification', default=False, action='store_true')
parser.add_argument('-oracle','--oracle', help='Fill slot values from ground-truth', default=False, action='store_true')
parser.add_argument('-data','--data', help='Dataset (mwz/snips)', type=str, default='mwz', required=False, choices=['mwz', 'snips'])
parser.add_argument('-dev','--dev', help='Generate output for dev set instead of test set', default=False, action='store_true')

args = vars(parser.parse_args())
model_dir = args['path']
model_name = args["model"]
best_model_ckpt =  args['best']
c_wt = args['m']
use_classification_head = args['slot']
use_oracle = args['oracle']
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
    print("Your model name is not Llama/Gemma/Mistral. Set model-specific prompt and hidden size.")

if(is_dev):
    result_file = os.path.join(model_dir, "result_output_dev.json")
else:
    result_file = os.path.join(model_dir, "result_output.json")
if(use_oracle):
    fn = "dst_oraclecorrect_dev.txt" if is_dev else "dst_oraclecorrect.txt"
    out_file = os.path.join(model_dir, fn)
else:
    fn = "dst_selfcorrect_dev.txt" if is_dev else "dst_selfcorrect.txt"
    out_file = os.path.join(model_dir, fn)

fuzzy_threshold = 95

print(f"Result file: {result_file}")
print(f"model_name: {model_name}")
print(f"best_model_ckpt: {best_model_ckpt}")
print(f"use_classification_head: {use_classification_head}")
print(f"use_oracle: {use_oracle}")
print(f"Output file: {out_file}")
print(f"dataset_name: {dataset_name}")
print(f"is_dev: {is_dev}")

with open(result_file, "r") as f:
    result = json.load(f)

if(dataset_name=="snips"):
    domains, slot_list, slot_details, dom_slot_dict = get_snips_details()
else:
    slot_list = get_slot_list()

f = open(out_file, "w")

#------------------------------------------------

# Model settings
if(not use_oracle):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print(f"Using GPU!")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

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

    hidden_size = 4096
    if(model_type=="gemma"):
        hidden_size = 3072
    num_class = len(slot_list)

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

    # Check the path of best model checkpoint
    if(not os.path.isdir(best_model_ckpt)):
        print(f"Model checkpoint does not exist. Exiting the code.")
        exit(0)

    model = DstModel.from_pretrained(best_model_ckpt)
    model.to(device)
    model.eval()
    print("Best model loaded")

#---------------------

slot_dict = {}
for i in range(len(slot_list)):
    slot_dict[slot_list[i]] = i

def is_match(bs_gt, bs_pr):
    f_match = True
    is_fn = False
    is_fp = False
    is_ve = False

    key1 = set(list(bs_gt.keys()))
    key2 = set(list(bs_pr.keys()))

    s1 = key1.difference(key2)
    s2 = key2.difference(key1)

    if(len(s1)==0 and len(s2)==0):
        if(len(bs_gt)>0):
            for sk in bs_gt:
                v1 = bs_gt[sk]
                v2 = bs_pr[sk]
                if(fuzz.partial_ratio(v1, v2)<=fuzzy_threshold):
                    f_match = False
                    is_ve = True
                    break
    else:
        f_match = False
        if(len(s1)>0):
            is_fn = True
        if(len(s2)>0):
            is_fp = True
    c = 1 if f_match else 0
    return c, is_fn, is_fp, is_ve

def get_slot_match(bs_gt, bs_pred):
    y_gt = []
    y_pr = []
    for slot in slot_list:
        y1 = 0
        y2 = 0
        if(slot in bs_gt and slot not in bs_pred):
            y1 = 1
            y2 = 0
        elif(slot not in bs_gt and slot in bs_pred):
            y1 = 0
            y2 = 1
        elif(slot in bs_gt and slot in bs_pred):
            y1 = 1
            y2 = 1
            if(fuzz.partial_ratio(bs_gt[slot], bs_pred[slot])<=fuzzy_threshold):
                y2 = 0
        
        y_gt.append(y1)
        y_pr.append(y2)

    return y_gt, y_pr

def get_confidence_scores(bs_gt, bs_pr, slot_prob):
    lst_lbl_conf = []
    lst_score_conf = []
    for slot in bs_gt:
        if slot not in bs_pr:
            lst_lbl_conf.append(0)
        else:
            if(fuzz.partial_ratio(bs_gt[slot], bs_pr[slot])<=fuzzy_threshold):
                lst_lbl_conf.append(0)
            else:
                lst_lbl_conf.append(1)
        lst_score_conf.append(slot_prob[slot])   

    for slot in bs_pr:
        if slot not in bs_gt:
            lst_lbl_conf.append(0)
            lst_score_conf.append(1.0-slot_prob[slot])
            
    return lst_lbl_conf,lst_score_conf

def filter_false_positives(bs_pred, bs_gt, slot_prob, fp_threshold):
    bs_pred2 = {}
    for slot in bs_pred:
        if(slot in slot_prob and slot_prob[slot]>=fp_threshold):
            bs_pred2[slot] = bs_pred[slot]
        else:
            if(use_oracle and slot_prob[slot]<fp_threshold and slot in bs_gt):
                bs_pred2[slot] = bs_pred[slot]

    return bs_pred2

def generate_slot_value(utt, prompt_bs_pr, slot):
    if(dataset_name=="snips"):
        sl = slot.split("-")[1]
        prompt_bs_pr = prompt_bs_pr.replace("}}", "")
        prompt_tmp = prompt_bs_pr + f", \"{sl}\": \""
        prompt = get_prompt_snips(utt, domains, slot_details, model_type, tokenizer)
    else:
        prompt_bs_pr = prompt_bs_pr.replace("}", "")
        prompt_tmp = prompt_bs_pr + f", \"{slot}\": \""
        prompt = get_prompt(utt, model_type, tokenizer)
    prompt = prompt + prompt_tmp

    slot_val = ""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    idx = input_ids.size(1)
    with torch.no_grad():
        output = model.lm.generate(input_ids=input_ids, max_new_tokens=32)
        resp_idx = output[0][idx:]
        val = tokenizer.decode(resp_idx, skip_special_tokens=True)
        slot_val = val.split("\"")[0].strip()
    return slot_val

def add_false_negatives(bs_pred, slot_prob, fn_threshold, bs_gt, utt, prompt_bs_pr):
    e_calls = 0
    bs_pred2 = copy.deepcopy(bs_pred)
    for slot in slot_prob:
        if(slot_prob[slot]>=fn_threshold and slot not in bs_pred2):
            if(use_oracle):
                if(slot in bs_gt):
                    bs_pred2[slot] = bs_gt[slot]
                    e_calls+=1
            else:
                val = generate_slot_value(utt, prompt_bs_pr, slot)
                if(len(val)>0 and val!="none"):
                    bs_pred2[slot] = val
                e_calls+=1

    return bs_pred2, e_calls

def filter_fpfn_joint(bs_pred, slot_prob, fn_threshold, fp_threshold, bs_gt, utt, prompt_bs_pr):
    e_calls=0
    bs_pred2 = {}
    for slot in bs_pred:
        if(slot in slot_prob and slot_prob[slot]>=fp_threshold):
            bs_pred2[slot] = bs_pred[slot]
        else:
            if(use_oracle and slot_prob[slot]<fp_threshold and slot in bs_gt):
                bs_pred2[slot] = bs_pred[slot]
            
    for slot in slot_prob:
        if(slot_prob[slot]>=fn_threshold and slot not in bs_pred2):
            if(use_oracle):
                if(slot in bs_gt):
                    bs_pred2[slot] = bs_gt[slot]
                    e_calls+=1
            else:
                val = generate_slot_value(utt, prompt_bs_pr, slot)
                if(len(val)>0 and val!="none"):
                    bs_pred2[slot] = val
                e_calls+=1
    return bs_pred2, e_calls

def get_fp_fn(bs_gt, bs_pr):
    is_fn = False
    is_fp = False
    key1 = set(list(bs_gt.keys()))
    key2 = set(list(bs_pr.keys()))

    s1 = key1.difference(key2)
    s2 = key2.difference(key1)
    if(len(s1)>0):
        is_fn = True
    if(len(s2)>0):
        is_fp = True

    return is_fn,is_fp,s1,s2

def check_slots(bs):
    bs_new = {}
    for slot in bs:
        if slot in slot_list:
            bs_new[slot] = bs[slot]
    return bs_new

def modify_snips_bs(bs):
    bs_new = {}
    dom = ""
    if "domain" in bs:
        dom = bs["domain"]
    slot_values = {}
    if "slot_values" in bs:
        slot_values = bs["slot_values"]
    for slot in slot_values:
        k = f"{dom}-{slot}"
        bs_new[k] = slot_values[slot]
    return bs_new

#--------------------------
#Evaluation of Original Prediction

total = 0
c_correct = 0
c_error = 0
parse_err = 0
c_fpfn_original = {"fn":0, "fp":0, "ve":0}
lst_label = []
lst_prob = []
y_true = []
y_pred = []
    
for res in result:
    total+=1
    utt = res["utt"]
    bs_gt = res["bs_gt"]
    if(dataset_name=="snips"):
        bs_gt = modify_snips_bs(bs_gt)
    bs_pred = res["bs_pred"]
    try:
        bs_pr = json.loads(bs_pred)
        if(dataset_name=="snips"):
            bs_pr = modify_snips_bs(bs_pr)

        bs_pr = check_slots(bs_pr)
        match, is_fn, is_fp, is_ve = is_match(bs_gt, bs_pr)

        if(is_fn or is_fp or is_ve):
            c_error+=1
        else:
            c_correct+=1
        if(is_fn):
            c_fpfn_original["fn"] = c_fpfn_original["fn"] + 1
        if(is_fp):
            c_fpfn_original["fp"] = c_fpfn_original["fp"] + 1
        if(is_ve):
            c_fpfn_original["ve"] = c_fpfn_original["ve"] + 1
            
        y_gt, y_pr = get_slot_match(bs_gt, bs_pr)
        y_true.extend(y_gt)
        y_pred.extend(y_pr)

        if(use_classification_head):
            slot_prob = res["prob"]
            lst_lbl_conf, lst_score_conf = get_confidence_scores(bs_gt, bs_pr, slot_prob)
            lst_label.extend(lst_lbl_conf)
            lst_prob.extend(lst_score_conf) 
            
    except Exception as e:
        f.write(f"Error: {bs_pred}: {e}\n")
        f.write("-"*30 + "\n")
        parse_err+=1

print(f"#Parsing error: {parse_err}")
f.write("-"*30 + "\n")
f.write(f"#Total: {total}\n")
f.write(f"#Correct: {c_correct}\n")
f.write(f"#Mistake: {c_error}\n")
f.write(f"#Parsing error: {parse_err}\n")
f.write(f"JGA: {(c_correct*100.0)/total}\n")
print(f"JGA: {(c_correct*100.0)/total}")
f.write(f"Errors: {c_fpfn_original}\n")
print(f"Errors: {c_fpfn_original}")
f.write(f"FPR: {c_fpfn_original["fp"]*100.0/total}, FNR: {c_fpfn_original["fn"]*100.0/total}\n")
f.write(f"Slot F1 Score: {f1_score(y_true, y_pred)}\n")
f.write(f"Slot F1 Score check len: {len(y_true)} {len(y_pred)}\n")
print(f"Slot F1 Score: {f1_score(y_true, y_pred)}")
if(use_classification_head):
    roc_auc = roc_auc_score(lst_label, lst_prob)
    f.write(f"roc_auc: {roc_auc}\n")
    print(f"roc_auc: {roc_auc}")

f.write("-"*30 + "\n")

if(not use_classification_head):
    print("done")
    exit(0)

#--------------------------
#Evaluation after removing False Positives

f.write("Correction using Filtering False Positives::\n")
f.write("-"*30 + "\n")
lst_fp_threshold = [0.05, 0.1, 0.2, 0.3]

for fp_threshold in lst_fp_threshold:
    p_err = 0
    c_correct = 0
    c_error = 0
    c_total = 0
    c_fpfn_filter = {"fn":0, "fp":0, "ve":0}
    lst_label = []
    lst_prob = []
    y_true = []
    y_pred = []

    for res in result:
        c_total+=1
        utt = res["utt"]
        bs_gt = res["bs_gt"]
        if(dataset_name=="snips"):
            bs_gt = modify_snips_bs(bs_gt)
        bs_pred = res["bs_pred"]
        try:
            bs_pr = json.loads(bs_pred)
            if(dataset_name=="snips"):
                bs_pr = modify_snips_bs(bs_pr)

            bs_pr = check_slots(bs_pr)
            slot_prob = res["prob"]
            bs_pr_filter = filter_false_positives(bs_pr, bs_gt, slot_prob, fp_threshold)
            match, is_fn, is_fp, is_ve = is_match(bs_gt, bs_pr_filter)
            if(is_fn or is_fp or is_ve):
                c_error+=1
            else:
                c_correct+=1
            if(is_fn):
                c_fpfn_filter["fn"] = c_fpfn_filter["fn"] + 1
            if(is_fp):
                c_fpfn_filter["fp"] = c_fpfn_filter["fp"] + 1
            if(is_ve):
                c_fpfn_filter["ve"] = c_fpfn_filter["ve"] + 1
                
            y_gt, y_pr = get_slot_match(bs_gt, bs_pr_filter)
            y_true.extend(y_gt)
            y_pred.extend(y_pr)   
            
            if(use_classification_head):
                slot_prob = res["prob"]
                lst_lbl_conf, lst_score_conf = get_confidence_scores(bs_gt, bs_pr_filter, slot_prob)
                lst_label.extend(lst_lbl_conf)
                lst_prob.extend(lst_score_conf)
            
        except Exception as e:
            f.write(f"Error: {bs_pred}: {e}\n")
            f.write("-"*30 + "\n")
            p_err+=1
    
    f.write(f"Result with fp threshold {fp_threshold}:-")
    f.write(f"#Total: {c_total}\n")
    f.write(f"#Correct: {c_correct}\n")
    f.write(f"#Mistake: {c_error}\n")
    f.write(f"JGA: {(c_correct*100.0)/c_total}\n")
    f.write(f"Errors: {c_fpfn_filter}\n")
    f.write(f"FPR: {c_fpfn_filter["fp"]*100.0/total}, FNR: {c_fpfn_filter["fn"]*100.0/total}\n")
    f.write(f"Slot F1 Score: {f1_score(y_true, y_pred)}\n")
    if(use_classification_head):
        roc_auc = roc_auc_score(lst_label, lst_prob)
        f.write(f"roc_auc: {roc_auc}\n")
    f.write("-"*20 + "\n")
f.write("-"*30 + "\n")

#---------------------------------
#Evaluation after adding False Negatives

f.write("Correction using Adding False Negatives::\n")
f.write("-"*30 + "\n")
lst_fn_threshold = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

for fn_threshold in lst_fn_threshold:
    p_err = 0
    c_correct = 0
    c_error = 0
    c_total = 0
    extra_check = []
    c_fpfn_filter = {"fn":0, "fp":0, "ve":0}
    lst_label = []
    lst_prob = []
    y_true = []
    y_pred = []

    for res in result:
        c_total+=1
        utt = res["utt"]
        bs_gt = res["bs_gt"]
        if(dataset_name=="snips"):
            bs_gt = modify_snips_bs(bs_gt)
        bs_pred = res["bs_pred"]
        try:
            bs_pr = json.loads(bs_pred)
            if(dataset_name=="snips"):
                bs_pr = modify_snips_bs(bs_pr)

            bs_pr = check_slots(bs_pr)
            slot_prob = res["prob"]
            bs_pr_filter, e_calls = add_false_negatives(bs_pr, slot_prob, fn_threshold, bs_gt, utt, res["bs_pred"])
            if(e_calls>0):
                extra_check.append(e_calls)
            match, is_fn, is_fp, is_ve = is_match(bs_gt, bs_pr_filter)
            if(is_fn or is_fp or is_ve):
                c_error+=1
            else:
                c_correct+=1
            if(is_fn):
                c_fpfn_filter["fn"] = c_fpfn_filter["fn"] + 1
            if(is_fp):
                c_fpfn_filter["fp"] = c_fpfn_filter["fp"] + 1
            if(is_ve):
                c_fpfn_filter["ve"] = c_fpfn_filter["ve"] + 1
                
            y_gt, y_pr = get_slot_match(bs_gt, bs_pr_filter)
            y_true.extend(y_gt)
            y_pred.extend(y_pr)   
            
            if(use_classification_head):
                slot_prob = res["prob"]
                lst_lbl_conf, lst_score_conf = get_confidence_scores(bs_gt, bs_pr_filter, slot_prob)
                lst_label.extend(lst_lbl_conf)
                lst_prob.extend(lst_score_conf)

        except Exception as e:
            f.write(f"Error: {bs_pred}: {e}\n")
            f.write("-"*30 + "\n")
            p_err+=1
    
    f.write(f"Result with fn threshold {fn_threshold}:-")
    f.write(f"#Total: {c_total}\n")
    f.write(f"#Correct: {c_correct}\n") 
    f.write(f"#Mistake: {c_error}\n") 
    f.write(f"#Total extra check: {len(extra_check)}: {sum(extra_check)}\n")
    f.write(f"#Max extra check: {max(extra_check)}\n")
    f.write(f"#Avg. extra check: {sum(extra_check)/len(extra_check)}\n")
    f.write(f"JGA: {(c_correct*100.0)/c_total}\n")
    f.write(f"Errors: {c_fpfn_filter}\n")
    f.write(f"FPR: {c_fpfn_filter["fp"]*100.0/total}, FNR: {c_fpfn_filter["fn"]*100.0/total}\n")
    f.write(f"Slot F1 Score: {f1_score(y_true, y_pred)}\n")
    if(use_classification_head):
        roc_auc = roc_auc_score(lst_label, lst_prob)
        f.write(f"roc_auc: {roc_auc}\n")
    f.write("-"*30 + "\n")
f.write("-"*30 + "\n")

#---------------------------------
#Evaluation after removing False Positives and adding False Negatives

f.write("Correction using both False Positive and False Negative::\n")
f.write("-"*30 + "\n")
lst_fn_threshold = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
lst_fp_threshold = [0.05, 0.1, 0.2, 0.3]
best_fn_threshold = 0
best_fp_threshold = 0
best_jga = 0

for fn_threshold in lst_fn_threshold:
    for fp_threshold in lst_fp_threshold:
        p_err = 0
        c_correct = 0
        c_error = 0
        c_total = 0
        extra_check = []
        c_fpfn_filter = {"fn":0, "fp":0, "ve":0}
        lst_label = []
        lst_prob = []
        y_true = []
        y_pred = []

        for res in result:
            c_total+=1
            utt = res["utt"]
            bs_gt = res["bs_gt"]
            if(dataset_name=="snips"):
                bs_gt = modify_snips_bs(bs_gt)
            bs_pred = res["bs_pred"]
            try:
                bs_pr = json.loads(bs_pred)
                if(dataset_name=="snips"):
                    bs_pr = modify_snips_bs(bs_pr)

                bs_pr = check_slots(bs_pr)
                slot_prob = res["prob"]
                bs_pr_filter, e_calls = filter_fpfn_joint(bs_pr, slot_prob, fn_threshold, fp_threshold, bs_gt, utt, res["bs_pred"])
                if(e_calls>0):
                    extra_check.append(e_calls)
                match, is_fn, is_fp, is_ve = is_match(bs_gt, bs_pr_filter)
                if(is_fn or is_fp or is_ve):
                    c_error+=1
                else:
                    c_correct+=1
                #c_correct+=match
                if(is_fn):
                    c_fpfn_filter["fn"] = c_fpfn_filter["fn"] + 1
                if(is_fp):
                    c_fpfn_filter["fp"] = c_fpfn_filter["fp"] + 1
                if(is_ve):
                    c_fpfn_filter["ve"] = c_fpfn_filter["ve"] + 1
                    
                y_gt, y_pr = get_slot_match(bs_gt, bs_pr_filter)
                y_true.extend(y_gt)
                y_pred.extend(y_pr)   
                
                if(use_classification_head):
                    slot_prob = res["prob"]
                    lst_lbl_conf, lst_score_conf = get_confidence_scores(bs_gt, bs_pr_filter, slot_prob)
                    lst_label.extend(lst_lbl_conf)
                    lst_prob.extend(lst_score_conf)
                    
            except Exception as e:
                p_err+=1
        
        f.write(f"Result with fn threshold {fn_threshold} and fp threshold {fp_threshold} and:-")
        f.write(f"#Total: {c_total}\n")
        f.write(f"#Correct: {c_correct}\n")
        f.write(f"#Mistake: {c_error}\n")
        f.write(f"#Total extra check: {len(extra_check)}: {sum(extra_check)}\n")
        f.write(f"#Max extra check: {max(extra_check)}\n")
        f.write(f"#Avg. extra check: {sum(extra_check)/len(extra_check)}\n")

        jga = (c_correct*100.0)/c_total
        f.write(f"JGA: {jga}\n")
        if(jga>best_jga):
            best_fn_threshold = fn_threshold
            best_fp_threshold = fp_threshold
            best_jga = jga

        f.write(f"Errors: {c_fpfn_filter}\n")
        f.write(f"FPR: {c_fpfn_filter["fp"]*100.0/total}, FNR: {c_fpfn_filter["fn"]*100.0/total}\n")
        f.write(f"Slot F1 Score: {f1_score(y_true, y_pred)}\n")
        if(use_classification_head):
            roc_auc = roc_auc_score(lst_label, lst_prob)
            f.write(f"roc_auc: {roc_auc}\n")
        f.write("-"*30 + "\n")
    f.write("-"*30 + "\n")
f.write("-"*30 + "\n")

f.write("-"*30 + "\n")
f.write(f"Best JGA={best_jga} with threshold : fn = {best_fn_threshold}, fp = {best_fp_threshold}\n")
f.close()

print("done")

#---------------------------------
