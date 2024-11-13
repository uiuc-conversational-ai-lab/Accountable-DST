import json
import os

def save_json(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

domains = set()
slots = set()
slot_dict = {}
filename = os.path.join("snips", "train.json")

with open(filename, 'r') as f:
    data = json.load(f)

for i in range(len(data)):
    dom = data[i]["domain"]
    domains.add(dom)
    if(dom not in slot_dict):
        slot_dict[dom] = set()
    for slot in data[i]["slot_values"]:
        slot_dict[dom].add(slot)
        slots.add(f"{dom}-{slot}")

slot_dict_final = {}
for dom in slot_dict:
    slot_dict_final[dom] = list(slot_dict[dom])
        
save_json(list(domains), os.path.join("snips", "domains.json"))
save_json(list(slots), os.path.join("snips", "slots.json"))
save_json(slot_dict_final, os.path.join("snips", "slot_dict.json"))