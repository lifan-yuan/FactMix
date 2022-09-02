import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pathlib import Path
import re
from torch.utils.data import DataLoader
import os
import pandas as pd
import datasets
import copy
import random
from typing import List
from collections import defaultdict
from sklearn import metrics

# ckpt = 'bert-base-cased'
# tokenizer = AutoTokenizer.from_pretrained(ckpt, add_prefix_space=True)
global ckpt
global tokenizer
global device
global tag2id
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# tag2id = {'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}



def read_data(DataPath, task):
    
    if not os.path.exists(os.path.join(DataPath, 'test')):

        spliter = ' '
        splits = ['train', 'dev', 'test'] if task == 'conll2003' else ['test']

        for split in splits:
            SplitPath = os.path.join(DataPath, split)
            dataset = {'tokens':[], 'tags':[]}
            
            DataPth = Path(SplitPath+'.txt')
            raw_texts = DataPth.read_text().strip()
            raw_docs = raw_texts.split('\n\n') # a doc of sentences

            token_docs = []
            tag_docs = []
            # random.shuffle(raw_docs)
            
            for i, sent in enumerate(raw_docs):
                sent = sent.split('\n')
                token_list = []
                tag_list = []
                for line in sent:
                    line = line.split(spliter)
                    if len(line) < 2:
                        print(DataPath, task, sent, line)
                        continue
                    # print(line)
                    token, tag = line[0], line[1]
                    token_list.append(token)
                    tag_list.append(tag)

                dataset['tokens'].append(token_list)
                dataset['tags'].append(tag_list)

            dataset = pd.DataFrame(dataset)
            dataset = datasets.Dataset.from_pandas(dataset)
            dataset.save_to_disk(SplitPath)
    
    if task == 'conll2003':
        train = load_from_disk(os.path.join(DataPath, 'train'))
        dev = load_from_disk(os.path.join(DataPath, 'dev'))
    else:
        train = None
        dev = None
    test = load_from_disk(os.path.join(DataPath, 'test'))

    return train, dev, test

def sampling(data, label_list, tag2id, num_samples=50): # num_samples per tag
    if num_samples == -1:
        return data
    count = np.zeros((len(label_list),), dtype=np.int64)
    sampled_data = []

    for d in data:
        idx = np.where(count<num_samples)[0] 
        id_list = [tag2id[tag] for tag in d['tags']]
        
        if len(set(idx).intersection(set(id_list))) == 0: # There is no currently required tag
            continue
        sampled_data.append(d)

        for label in d['tags']:
            count[tag2id[label]] += 1

    return sampled_data



def collate_fn(input_pad_id, output_pad_token, output_pad_id, device):

    def collate_fn_wrapper(batch):
        max_seq_len = 36
        # truncation
        for i, _ in enumerate(batch):
            if len(batch[i]["input_ids"]) > max_seq_len:
                batch[i]["input_ids"] = batch[i]["input_ids"][:max_seq_len]
                batch[i]["labels"] = batch[i]["labels"][:max_seq_len]
                batch[i]["attention_mask"] = batch[i]["attention_mask"][:max_seq_len]
        # padding
        for i, _ in enumerate(batch):
            length = len(batch[i]["input_ids"])
            batch[i]["input_ids"] += [input_pad_id] * (max_seq_len - length) 
            batch[i]["labels"] += [output_pad_id] * (max_seq_len - length) 
            batch[i]["attention_mask"] = [1] * length + [0] * (max_seq_len - length)

        input_ids = torch.LongTensor([sample['input_ids'] for sample in batch]).to(device)
        labels = torch.LongTensor([sample['labels'] for sample in batch]).to(device)
        masks = torch.LongTensor([sample['attention_mask'] for sample in batch]).to(device)
        return (input_ids, labels, masks)
            
    return collate_fn_wrapper


def tokenize_aug(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    examples["tag_ids"] = [[tag2id[tag] for tag in tags] for tags in examples["tags"]]
    for i, label in enumerate(examples["tag_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def comp(ori_model, sampled_trainset, reasonable_aug_examples, batch_size, 
        input_pad_id, output_pad_id, output_pad_token, aug_num=None):
    
    ori_loader = DataLoader(sampled_trainset, batch_size=batch_size, shuffle=False, 
                                    collate_fn=collate_fn(input_pad_id, output_pad_token, output_pad_id, device))
    gen_loader = DataLoader(reasonable_aug_examples, batch_size=batch_size, shuffle=False, 
                                    collate_fn=collate_fn(input_pad_id, output_pad_token, output_pad_id, device))
    ori_hidden = []
    gen_hidden = []
    for batch in ori_loader:
        input_ids, bathch_labels, masks = batch
        outputs = ori_model(input_ids=input_ids, attention_mask=masks, output_hidden_states=True)
        ori_hidden.extend(outputs.hidden_states[0].detach().cpu().tolist())

    for batch in gen_loader:
        input_ids, bathch_labels, masks = batch
        outputs = ori_model(input_ids=input_ids, attention_mask=masks, output_hidden_states=True)
        gen_hidden.extend(outputs.hidden_states[0].detach().cpu().tolist())


    ori_gen_pair = []
    last_ori_sent = None
    for i, example in enumerate(reasonable_aug_examples):
        ori_index = example['ori_index']
        ori_h = np.array(ori_hidden[ori_index])
        gen_h = np.array(gen_hidden[i])
        ori_gen_pair.append((ori_h, gen_h))

    gen_rank = []
    for (ori, gen) in ori_gen_pair:
        ori, gen = np.array(ori), np.array(gen)
        NMI = metrics.normalized_mutual_info_score(ori.reshape((-1,)), gen.reshape((-1,)))
        gen_rank.append(NMI)
    # get the sorted index of the element in the original array
    # the index of smaller NMI scores will be presented first
    # i.e., in variable 'order', we should retain last aug_num samples for a larger NMI score
    order = np.argsort(np.array(gen_rank))

    return order


def create_counterfactual_examples(trainset, pad_tag, aug_num=50):
    deduplicated_examples = set()
    counterfactual_examples = []
    local_entity_sets = {}

    for example in trainset:
        deduplicated_examples.add(copy.deepcopy(' '.join(example["tokens"])))
        for token, tags in zip(example["tokens"], example['tags']):
            if tags == pad_tag:
                continue
            if tags in local_entity_sets.keys():
                local_entity_sets[tags].append(token)
            else:
                local_entity_sets[tags] = [token]
    for key in local_entity_sets.keys():
            local_entity_sets[key] = list(set(local_entity_sets[key]))
    local_entity_sets[pad_tag] = []
    for i, example in enumerate(trainset):
        local_entity_subsets = []
        count = 0
        while len(local_entity_subsets) < 1 and count < len(example['tokens']):
            index = random.choice(list(range(len(example['tokens']))))
            count += 1
            if len(local_entity_sets[example["tags"][index]]) > 0:
                local_entity_subsets = local_entity_sets[example["tags"][index]]
            

        for j, local_candidate in enumerate(local_entity_subsets):
            cfexample = copy.deepcopy(example)
            cfexample["ori_index"] = i
            cfexample["obersavational_text"] = example["tokens"]
            cfexample["tokens"][index] = local_candidate
            if cfexample["tokens"] == example["tokens"] or ' '.join(cfexample["tokens"]) in deduplicated_examples:
                continue

            deduplicated_examples.add(copy.deepcopy(' '.join(cfexample["tokens"])))
            cfexample["replaced"] = [
                "[{0}]({1}, {2})".format(
                    cfexample["tokens"][index], index, example["tags"][index].split('-')[1]
                )
            ]
            counterfactual_examples.append(cfexample)
            # if j % aug_num == 0:
            #     print(j)
    return counterfactual_examples, deduplicated_examples


def create_semifactual_examples(trainset, pad_tag, aug_num=50):
    deduplicated_examples = set()
    counterfactual_examples = []
    # filler = SubstituteWithBert()
    filler = pipeline('fill-mask', model=ckpt, tokenizer=ckpt, device=0)
    for i, example in enumerate(trainset):

        for index in range(len(example['tokens'])):
            if example['tags'][index] != pad_tag: # we only substitute the Non-O token
                continue

            cfexample = copy.deepcopy(example)
            cfexample["ori_index"] = i
            cfexample["obersavational_text"] = example["tokens"]
            # mask-and-fill
            # cfexample["input_ids"][index+1] = 103 # [MASK] ~ 103, first place of input_ids is [CLS]
            cfexample["tokens"][index] = '[MASK]'
            fill_result = filler(' '.join(cfexample["tokens"]), top_k=1)
            candidate = fill_result[0]['token_str']
            cfexample["tokens"][index] = candidate
            
            if cfexample["tokens"] == example["tokens"] or ' '.join(cfexample["tokens"]) in deduplicated_examples:
                continue

            deduplicated_examples.add(copy.deepcopy(' '.join(cfexample["tokens"])))
            cfexample["replaced"] = [
                "[{0}]({1}, {2})".format(
                    cfexample["tokens"][index], index, example["tags"][index]
                )
            ]
            counterfactual_examples.append(cfexample)
            # if j % aug_num == 0:
            #     print(j)
    return counterfactual_examples, deduplicated_examples




def check_data(model, all_aug_examples, aug_dataloader, id2tag, batch_size):
    reasonable_aug_examples, unreasonable_aug_examples = [], []
    # check if reasonable
    for i, batch in enumerate(aug_dataloader):
        input_ids, labels, masks = batch # 32 sents
        outputs = model(input_ids=input_ids, attention_mask=masks)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, -1)
        preds = probs.argmax(-1).detach().cpu().tolist()
        for j, pred in enumerate(preds): # 32 sents, pred ~ 1 sent
            tokens = all_aug_examples[i*batch_size + j]["tokens"]
            tags = all_aug_examples[i*batch_size + j]["tags"]
            replaced_spans = all_aug_examples[i*batch_size + j]["replaced"]
            predicted_spans = ["[{0}]({1}, {2})".format(token, index, id2tag[pred_id].split('-')[1] if id2tag[pred_id] != 'O' else 'O') 
                                for index, (token, pred_id) in enumerate(zip(tokens, pred))]
            if len(set(replaced_spans).intersection(set(predicted_spans))) == len(replaced_spans):
                reasonable_aug_examples.append(all_aug_examples[i*batch_size + j])
                # print('add')
    reasonable_aug_examples = pd.DataFrame(reasonable_aug_examples)
    reasonable_aug_examples = datasets.Dataset.from_pandas(reasonable_aug_examples)
    
    return reasonable_aug_examples, unreasonable_aug_examples


def augment(ori_model, sampled_trainset, myckpt, mytokenizer,
            input_pad_id, output_pad_id, output_pad_token, mytag2id, id2tag, 
            batch_size, aug_example_path, aug_ratio, mydevice, is_semi=False):
    
    global ckpt
    ckpt = myckpt
    global tokenizer
    tokenizer = mytokenizer
    global device
    device = mydevice
    global tag2id
    tag2id = mytag2id
    
    if os.path.exists(aug_example_path):
        print('loading reasonable cfexamples')
        selected_aug_examples = load_from_disk(aug_example_path)
    else:
        if aug_ratio == 0:
            return []
        if is_semi:
            all_aug_examples, deduplicated_examples = create_semifactual_examples(sampled_trainset, output_pad_token)
        else:
            all_aug_examples, deduplicated_examples = create_counterfactual_examples(sampled_trainset, output_pad_token)
        all_aug_examples = pd.DataFrame(all_aug_examples)
        all_aug_examples = datasets.Dataset.from_pandas(all_aug_examples)
        all_aug_examples = all_aug_examples.map(tokenize_aug, batched=True)
        # to check if the generated cf examples are linguistically reasonable
        aug_dataloader = DataLoader(all_aug_examples, batch_size, shuffle=True, collate_fn=collate_fn(input_pad_id, output_pad_token, output_pad_id, device))
        reasonable_aug_examples, unreasonable_aug_examples = check_data(ori_model, all_aug_examples, aug_dataloader, id2tag, batch_size)
        print(f'{len(sampled_trainset)=}')
        print(f'{len(all_aug_examples)=}')
        print(f'{len(reasonable_aug_examples)=}')
        print('maximum ratio:', int(len(reasonable_aug_examples) / len(sampled_trainset)))
        # return reasonable_aug_examples
        maximum = int(len(reasonable_aug_examples) / len(sampled_trainset))
        # sampling the augmented examples using MMI
        sorting_index = comp(ori_model, sampled_trainset, reasonable_aug_examples, batch_size, input_pad_id, output_pad_id, output_pad_token)
        
        for aug_ratio in range(1, maximum+1):
            aug_num = aug_ratio * len(sampled_trainset)
            aug_num = int(aug_num) if aug_num < len(reasonable_aug_examples) else len(reasonable_aug_examples)
            print('ori samples:', len(sampled_trainset), 'aug samples:', aug_num)
            path = aug_example_path.replace('aug_ratio', str(aug_ratio))
            sampling_index = sorting_index[-aug_num:]
            selected_aug_examples = reasonable_aug_examples.select(sampling_index)    
            remove_columns = set(selected_aug_examples.features) ^ set(sampled_trainset.features)
            selected_aug_examples = selected_aug_examples.remove_columns(remove_columns)
            selected_aug_examples.save_to_disk(path)
    
    aug_num = aug_ratio * len(sampled_trainset)
    sampling_index = sorting_index[-aug_num:]
    selected_aug_examples = reasonable_aug_examples.select(sampling_index)    
    
    return selected_aug_examples, maximum