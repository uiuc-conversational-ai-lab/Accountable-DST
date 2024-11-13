import json
import os
import copy
from datasets import Dataset, DatasetDict

max_token_limit = 900

#-------------------------------------
# MultiWOZ dataset

#Dataset path
train_file = os.path.join("mwz2.4", "train_dials.json")
dev_file = os.path.join("mwz2.4", "dev_dials.json")
test_file = os.path.join("mwz2.4", "test_dials.json")

#Multiwoz domain and slot details
domain_list = ["attraction", "hotel", "restaurant", "taxi", "train"]
attraction_slots = ["attraction-name", "attraction-type", "attraction-area"]
hotel_slots = ["hotel-name", "hotel-type", "hotel-parking", "hotel-area", "hotel-bookday", "hotel-bookstay", "hotel-internet", "hotel-bookpeople", "hotel-stars", "hotel-pricerange"]
restaurant_slots = ["restaurant-name", "restaurant-food", "restaurant-area", "restaurant-bookday", "restaurant-booktime", "restaurant-bookpeople", "restaurant-pricerange"]
taxi_slots = ["taxi-arriveby", "taxi-departure", "taxi-leaveat", "taxi-destination"]
train_slots = ["train-arriveby", "train-day", "train-leaveat", "train-destination", "train-departure", "train-bookpeople"]
slot_details_txt = f"The user can ask for a hotel by slots - {", ".join(hotel_slots)}. The user can ask for an attraction by slots - {", ".join(attraction_slots)}. The user can ask for a restaurant by slots - {", ".join(restaurant_slots)}. The user can ask for a taxi by slots - {", ".join(taxi_slots)}. The user can ask for a train by slots - {", ".join(train_slots)}. Do not capture any other slots!\n\n"

slot_dict = {}
lst_slots = []
lst_slots.extend(attraction_slots)
lst_slots.extend(hotel_slots)
lst_slots.extend(restaurant_slots)
lst_slots.extend(taxi_slots)
lst_slots.extend(train_slots)

for i in range(len(lst_slots)):
    slot_dict[lst_slots[i]] = i

#------------------------------------------------

def get_slot_list():
    return lst_slots

def getBeliefSet(bs):
    bs_new = {}
    for i in range(len(bs)):
        sv = bs[i]['slots']
        for j in range(len(sv)):
            slot_key = sv[j][0]
            slot_key = slot_key.replace("book ","book")
            slot_value = sv[j][1]
            if(slot_key in slot_dict):
                bs_new[slot_key] = slot_value
    return bs_new

def get_slot_label(bs):
    lbl_slots = [0]*len(slot_dict)
    for slot in bs:
        idx = slot_dict[slot]
        lbl_slots[idx] = 1
    return lbl_slots

def is_valid_domain(dt_mwz24):
    is_valid = True
    for j in range(len(dt_mwz24['dialogue'])):
        domain = dt_mwz24['dialogue'][j]['domain'].strip()
        if(len(domain)>0 and domain not in domain_list):
            is_valid = False
            break
    return is_valid
    

def load_mwz_data(filename, mode, test_run, conv_limit, logger, model_type, tokenizer):
    with open(filename, "r") as f:
        data_mwz24 = json.load(f)
    
    c_conv = 0
    c_turn = 0
    data_dict = {"utt": [], "bs": [], "slots": []}
    for i in range(len(data_mwz24)):
        dt_mwz24 = data_mwz24[i]
        if(is_valid_domain(dt_mwz24)):
            c_conv+=1
            conv_hist = ""
            for j in range(len(dt_mwz24['dialogue'])):
                sys = dt_mwz24['dialogue'][j]['system_transcript']
                usr = dt_mwz24['dialogue'][j]['transcript']
                
                if(model_type=="llama"):
                    turn_hist = f"<|start_header_id|>system<|end_header_id|>\n{sys.strip()}<|eot_id|>\n"
                    turn_hist = turn_hist + f"<|start_header_id|>user<|end_header_id|>\n{usr.strip()}<|eot_id|>\n"
                elif(model_type=="gemma"):
                    turn_hist = f"<start_of_turn>system\n{sys.strip()}<end_of_turn>\n"
                    turn_hist = turn_hist + f"<start_of_turn>user\n{usr.strip()}<end_of_turn>\n"
                else:
                    bos = tokenizer.bos_token
                    eos = tokenizer.eos_token
                    turn_hist = f"{bos}system\n{sys.strip()}{eos}\n"
                    turn_hist = turn_hist + f"{bos}user\n{usr.strip()}{eos}\n"
                conv_hist = conv_hist + turn_hist
                
                bs = getBeliefSet(dt_mwz24['dialogue'][j]['belief_state'])
                lbl_slots = get_slot_label(bs) 
                tmp_hist = copy.deepcopy(conv_hist)

                add_sample = True
                if(mode=="train"):
                    inp = tokenizer.encode(tmp_hist)
                    if(len(inp)>max_token_limit): #Exclude long conversations from training to avoid memory error
                        add_sample = False
                    
                if(add_sample):
                    data_dict["utt"].append(tmp_hist)
                    data_dict["bs"].append(json.dumps(bs))
                    data_dict["slots"].append(lbl_slots)
                    c_turn+=1
            
        if(test_run and c_conv==conv_limit):
            break
    
    logger.info(f"{mode} data: total conversations = {c_conv} : total turns: {c_turn}")
    logger.info("-"*30)
    return data_dict
    
def get_mwz_dataset(test_run, conv_limit, logger, model_type, tokenizer, train_all, train_limit):    
    train_data = load_mwz_data(train_file, "train", test_run, conv_limit, logger, model_type, tokenizer)
    dev_data = load_mwz_data(dev_file, "dev", test_run, conv_limit, logger, model_type, tokenizer)

    dataset = DatasetDict()
    if(test_run or train_all):
        dataset["train"] = Dataset.from_dict(train_data)
    else:
        train_data = Dataset.from_dict(train_data).shuffle(seed=0)
        s_idx = [v for v in range(train_limit)]
        sample_train_data = train_data.select(s_idx)
        dataset["train"] = sample_train_data
    dataset["dev"] = Dataset.from_dict(dev_data)
    return dataset

def get_mwz_test_dataset(test_run, conv_limit, logger, model_type, tokenizer):    
    test_data = load_mwz_data(test_file, "test", test_run, conv_limit, logger, model_type, tokenizer)
    dataset = DatasetDict()
    dataset["test"] = Dataset.from_dict(test_data)
    return dataset

def get_mwz_dev_dataset(test_run, conv_limit, logger, model_type, tokenizer):    
    test_data = load_mwz_data(test_file, "dev", test_run, conv_limit, logger, model_type, tokenizer)
    dataset = DatasetDict()
    dataset["test"] = Dataset.from_dict(test_data)
    return dataset

def get_prompt(conv_history, model_type, tokenizer):
    if(model_type=="mistral"):
        prompt = "[INST]"
    else:
        prompt = ""
    prompt = prompt + "You are a helpful assistant who can perform dialogue-state tracking. The user interacts with the system to book entities from multiple domains (hotel, restaurant, attraction, taxi, and train) in Cambridge. Your goal is to find all the intents shown by the user in the conversation.\n\n"
    prompt = prompt + slot_details_txt
    prompt = prompt + "# Task\nYou will be provided with a chronological dialogue history between the system and the user. You must find all the user intents and output them in JSON format.\n\n"
    prompt = prompt + """# Sample Output\n{"restaurant-name": "abc", "restaurant-food": "xyz"}\n\n"""
    prompt = prompt + f"# Conversation History\n"
    prompt = prompt + f"{conv_history}\n"
    if(model_type=="llama"):
        prompt = prompt + f"<|start_header_id|>assistant<|end_header_id|>"
    elif(model_type=="mistral"):
        prompt = prompt + "[/INST]"
    else:
        prompt = prompt + tokenizer.eos_token
    return prompt

#-------------------------------------
# SNIPS dataset

#Load SNIPS domain and slot details
def get_snips_details():
    filename = os.path.join("snips", "domains.json")
    with open(filename, 'r') as f:
        domains = json.load(f)

    filename = os.path.join("snips", "slots.json")
    with open(filename, 'r') as f:
        lst_slots = json.load(f)

    filename = os.path.join("snips", "slot_dict.json")
    with open(filename, 'r') as f:
        slot_dict = json.load(f)
            
    slot_details = ""
    for dom in slot_dict:
        slot_details = slot_details + f"The user can seek for {dom} by slots - " +  ", ".join(slot_dict[dom]) + ".\n"
    slot_details = slot_details + "Do not capture any other slots!\n\n"
    
    dom_slot_dict = {}
    for i in range(len(lst_slots)):
        dom_slot_dict[lst_slots[i]] = i
    
    return domains, lst_slots, slot_details, dom_slot_dict

def get_slot_label_snips(dom_slot_dict, bs):
    lbl_slots = [0]*len(dom_slot_dict)
    dom = bs['domain']
    for slot in bs['slot_values']:
        k = f"{dom}-{slot}"
        lbl_slots[dom_slot_dict[k]] = 1
    return lbl_slots

def load_snips_data(mode, slots, dom_slot_dict, test_run, conv_limit, logger):
    filename = os.path.join("snips", f"{mode}.json")
    with open(filename, "r") as f:
        data = json.load(f)
    c_conv = 0
    data_dict = {"utt": [], "bs": [], "slots": []}
    for i in range(len(data)):
        utt = data[i]['utterance']
        lbl = {}
        lbl['domain'] = data[i]['domain']
        lbl['slot_values'] = data[i]['slot_values']
        lbl_slots = get_slot_label_snips(dom_slot_dict, lbl)
        data_dict["utt"].append(utt)
        data_dict["bs"].append(json.dumps(lbl))
        data_dict["slots"].append(lbl_slots)
        c_conv+=1
        if(test_run and c_conv==conv_limit):
            break
    
    logger.info(f"{mode} data: total conversations = {c_conv}")
    logger.info("-"*30)
    return data_dict

def get_snips_dataset(test_run, conv_limit, logger, slots, dom_slot_dict):    
    train_data = load_snips_data("train", slots, dom_slot_dict, test_run, conv_limit, logger)
    dev_data = load_snips_data("valid", slots, dom_slot_dict, test_run, conv_limit, logger)

    dataset = DatasetDict()
    dataset["train"] = Dataset.from_dict(train_data)
    dataset["dev"] = Dataset.from_dict(dev_data)
    return dataset

def get_snips_infer_dataset(test_run, conv_limit, logger, slots, dom_slot_dict, is_dev):   
    if(is_dev):
        test_data = load_snips_data("valid", slots, dom_slot_dict, test_run, conv_limit, logger)
    else:
        test_data = load_snips_data("test", slots, dom_slot_dict, test_run, conv_limit, logger)
    dataset = DatasetDict()
    dataset["test"] = Dataset.from_dict(test_data)
    return dataset

def get_prompt_snips(utt, domains, slot_details_snips, model_type, tokenizer):
    if(model_type=="mistral"):
        prompt = "[INST]"
    else:
        prompt = ""
    prompt = prompt + "You are a helpful assistant who is assigned to find the intents shown by the user on 7 domains - " + ", ".join(list(domains)) + ".\n\n"
    prompt = prompt + slot_details_snips
    prompt = prompt + "# Task\nYou will be provided with an user utterance. You must find all the user intents and output them in JSON format.\n\n"
    prompt = prompt + """# Sample Output\n{"domain": "AddToPlaylist", "slot_values": {"music_item": "abc", "artist": "xyz"}}\n\n"""
    
    if(model_type=="llama"):
        prompt = prompt + f"<|start_header_id|>user<|end_header_id|>\n{utt.strip()}<|eot_id|>\n"
    elif(model_type=="gemma"):
        prompt = prompt + f"<start_of_turn>user\n{utt.strip()}<end_of_turn>\n"
    else:
        prompt = prompt + f"{tokenizer.bos_token}user\n{utt.strip()}{tokenizer.eos_token}\n"
        
    if(model_type=="llama"):
        prompt = prompt + f"<|start_header_id|>assistant<|end_header_id|>"
    elif(model_type=="mistral"):
        prompt = prompt + "[/INST]"
    else:
        prompt = prompt + tokenizer.eos_token
    return prompt

#-------------------------------------
    
