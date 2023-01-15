import argparse
import torch
from datasets import load_from_disk, concatenate_datasets, load_metric, load_dataset
from torch.utils.data import DataLoader
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import random
import numpy as np
import datasets
import pandas as pd
from process_data import *
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train(save_path, trainset, devset, data_collator, tokenizer, compute_metrics):
    if load and os.path.exists(save_path):
        print('Loading from', save_path)
        model = AutoModelForTokenClassification.from_pretrained(save_path)
        load=True
    else:
        print('Traininig and saving to', save_path)
        model = AutoModelForTokenClassification.from_pretrained(ckpt, num_labels=len(label_list))
        load=False

    args = TrainingArguments(
        'model_cache',
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4*batch_size,
        num_train_epochs=3,
        weight_decay=0,
        logging_steps=1000000000000,
        save_steps=1000000000000,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=trainset,
        eval_dataset=devset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if not load:
        trainer.train()
        trainer.evaluate()
        os.makedirs(save_path, exist_ok=True)
        trainer.model.save_pretrained(save_path)
    return trainer



def eval(trainer, testset):
    predictions, labels, _ = trainer.predict(testset)

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default='bert-base-cased')
    parser.add_argument('--data_folder', type=str, default='NER_datasets')
    parser.add_argument('--train_task', default='conll2003', type=str)
    parser.add_argument('--eval_tasks', type=str, nargs='+', default=['conll2003', 'tech_news', 'ai', 'literature', 'music', 'politics', 'science'])
    parser.add_argument('--output_pad_token', type=str, default='O')
    parser.add_argument('--label_list', type=str, nargs='+', default=['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'])
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--cf_aug_ratio', type=int, default=-1) # set to -1 to search the best ratio
    parser.add_argument('--semi_aug_ratio', type=int, default=-1) # set to -1 to search the best ratio
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--load', type=bool, default=False) # whether to load model and augmented samples from cache
    args = parser.parse_args()

    

    # initialize the args
    seeds = args.seeds
    ckpt = args.ckpt
    ckpt = ckpt
    task = args.train_task
    eval_tasks = args.eval_tasks
    num_samples = args.num_samples
    cf_aug_ratio = args.cf_aug_ratio
    semi_aug_ratio = args.semi_aug_ratio
    output_pad_token = args.output_pad_token
    label_list = args.label_list
    batch_size = args.batch_size
    load = args.load

    task_samples = task + '_' + str(num_samples)
    DataPath = os.path.join(args.data_folder, task)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    metric = load_metric("my_seqeval.py")

    tokenizer = AutoTokenizer.from_pretrained(ckpt, add_prefix_space=True)
    model_name = ckpt.split("/")[-1]
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainset, devset, _ = read_data(DataPath, task)
    
    tag2id = {tag: id for id, tag in enumerate(label_list)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    print(f"{tag2id=}")
    print(f"{id2tag=}")
    
    fn_kwargs = {"tokenizer":tokenizer, "tag2id":tag2id}

    devset = devset.map(tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)
    input_pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    output_pad_id = tag2id[output_pad_token]
    

    max_semi_ratio = 1000000
    max_cf_ratio = 1000000

    semi_aug_ratio_list = [semi_aug_ratio] if semi_aug_ratio >= 0 else range(1, max_semi_ratio)
    cf_aug_ratio_list = [cf_aug_ratio] if cf_aug_ratio >= 0 else range(1, max_cf_ratio)

    
    for semi_aug_ratio in semi_aug_ratio_list:
        if semi_aug_ratio > max_semi_ratio:
            break
        for cf_aug_ratio in cf_aug_ratio_list:
            if cf_aug_ratio > max_cf_ratio:
                break
            for seed in range(seeds):

                # Sampling from the training dataset
                sampled_path = os.path.join('sampled_trainset', task_samples, str(seed))
                if load and os.path.exists(sampled_path):
                    sampled_trainset = load_from_disk(sampled_path)
                else:
                    sampled_trainset = sampling(trainset, label_list, tag2id, num_samples) 
                    sampled_trainset = pd.DataFrame(sampled_trainset)
                    sampled_trainset = datasets.Dataset.from_pandas(sampled_trainset)
                    os.makedirs(sampled_path)
                    sampled_trainset.save_to_disk(sampled_path)
                print(f"{len(sampled_trainset)=}")
                print(sampled_trainset)

                # Process data for the original model
                ori_dataset = sampled_trainset
                ori_train_data = ori_dataset.map(tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)
                
                # Train the original model under the few-shot setting
                print('\n'*2, 'Training ori model', '\n'*3)
                ori_save_path = f'ori_model/{ckpt}-{num_samples}/{seed}'
                ori_trainer = train(ori_save_path, ori_train_data, devset,
                        data_collator, tokenizer, compute_metrics)
                ori_model = ori_trainer.model
                

                # Generate semi-factual examples
                # Train semi-model with only semi-factual augmentation
                print('\n'*2, 'Training semi model', '\n'*3)
                if semi_aug_ratio > 0: # requires augmentation
                    # Train the improved model with semi augmentation
                    semiexample_path = os.path.join('semi_examples', task_samples, ckpt, 'aug_ratio', str(seed))
                    semi_save_path = f'semi_model/{ckpt}-{num_samples}-semi_{semi_aug_ratio}/{seed}'
                    if load and os.path.exists(semiexample_path.replace('aug_ratio', str(semi_aug_ratio))):
                        print('Loading semi examples')
                    else:
                        print('Semi augmentation')
                        # Augment the semi examples into the original dataset
                        max_semi_ratio = augment(ori_model, ori_dataset, ckpt, tokenizer, 
                                                                        input_pad_id, output_pad_id, output_pad_token, 
                                                                        tag2id, id2tag, batch_size, semiexample_path, 
                                                                        semi_aug_ratio, device, is_semi=True)
                    print(f"{semi_aug_ratio=}")
                    selected_semiexamples = load_from_disk(semiexample_path.replace('aug_ratio', str(semi_aug_ratio)))
                    
                    # Mix with the orginal dataset
                    semi_dataset = datasets.concatenate_datasets([ori_dataset, selected_semiexamples]).shuffle()
                    semi_train_data = semi_dataset.map(tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)

                    # Train the semi model with the semi-dataset
                    semi_trainer = train(semi_save_path, semi_train_data, devset,
                                    data_collator, tokenizer, compute_metrics)
                else:
                    # No augmentation
                    selected_semiexamples = sampled_trainset
                    semi_model = ori_model
                
                # Generate counterfactual examples
                # See paper:
                # "Counterfactual Generator: A Weakly-Supervised Method for Named Entity Recognition"
                # <https://aclanthology.org/2020.emnlp-main.590.pdf>
                # Train cf-model with only cf augmentation
                print('\n'*2, 'training cf model', '\n'*3)
                if cf_aug_ratio > 0:
                    # train the improved model with cf augmentation
                    cfexample_path = os.path.join('cf_examples', task_samples, ckpt, 'aug_ratio', str(seed))
                    cf_save_path = f'cf_model/{ckpt}-{num_samples}-cf_{cf_aug_ratio}/{seed}'
                    if load and os.path.exists(cfexample_path.replace('aug_ratio', str(cf_aug_ratio))):
                        print('loading cf examples')
                    else:
                        print('cf augmentation')
                        # augment the cf examples into the original dataset
                        max_cf_ratio = augment(ori_model, ori_dataset, ckpt, tokenizer, 
                                                                    input_pad_id, output_pad_id, output_pad_token, 
                                                                    tag2id, id2tag, batch_size, cfexample_path, 
                                                                    cf_aug_ratio, device, is_semi=False)
                    print(f"{cf_aug_ratio=}")
                    selected_cfexamples = load_from_disk(cfexample_path.replace('aug_ratio', str(cf_aug_ratio)))
                    
                    # Mix with the orginal dataset
                    cf_dataset = datasets.concatenate_datasets([ori_dataset, selected_cfexamples]).shuffle()
                    cf_train_data = cf_dataset.map(tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)

                    # Train the cf model with the cf-dataset
                    cf_trainer = train(cf_save_path, cf_train_data, devset,
                        data_collator, tokenizer, compute_metrics)
                else:
                    # No augmentation
                    selected_cfexamples = sampled_trainset
                    cf_model = ori_model
                
                
                print('\n'*2, 'training mix model', '\n'*3)

                # train a mix model with both semi- and cf- augmentation
                mix_save_path = f'mix_model/{ckpt}-{num_samples}/semi_{semi_aug_ratio}/cf_{cf_aug_ratio}/{seed}'

                # mix the orinal dataset, semi-augmented dataset and cf-augmented dataset
                mix_dataset = datasets.concatenate_datasets([ori_dataset, selected_semiexamples, selected_cfexamples]).shuffle()
                mix_train_data = mix_dataset.map(tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)

                # Train the mix model
                mix_trainer = train(mix_save_path, mix_train_data, devset,
                        data_collator, tokenizer, compute_metrics)
                
                

                print('begin evaluation')
                
                for eval_task in eval_tasks:
                    eval_path = os.path.join(args.data_folder, eval_task)
                    _, _, testset = read_data(eval_path, eval_task)
                    testset = testset.map(tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)
                    devset_results = pd.DataFrame(columns=['ori_precision', 'ori_recall', 'ori_f1', 'ori_acc',
                                                                'semi_precision', 'semi_recall', 'semi_f1', 'semi_acc',
                                                                'cf_precision', 'cf_recall', 'cf_f1', 'cf_acc',
                                                                'mix_precision', 'mix_recall', 'mix_f1', 'mix_acc'
                                                                    ])
                    testset_results = pd.DataFrame(columns=['ori_precision', 'ori_recall', 'ori_f1', 'ori_acc',
                                                                'semi_precision', 'semi_recall', 'semi_f1', 'semi_acc',
                                                                'cf_precision', 'cf_recall', 'cf_f1', 'cf_acc',
                                                                'mix_precision', 'mix_recall', 'mix_f1', 'mix_acc'
                                                                    ])
                    
                    os.makedirs(f'results/{ckpt}-{num_samples}/semi_{semi_aug_ratio}/cf_{cf_aug_ratio}', exist_ok=True)

                    if eval_task == task:
                        ori_devset_results = eval(ori_trainer, devset)
                        semi_devset_results = eval(semi_trainer, devset)
                        cf_devset_results = eval(cf_trainer, devset)
                        mix_devset_results = eval(mix_trainer, devset)
                        devset_results.loc[seed, :] = [ ori_devset_results['overall_precision'], ori_devset_results['overall_recall'], 
                                                ori_devset_results['overall_f1'], ori_devset_results['overall_accuracy'],
                                                semi_devset_results['overall_precision'], semi_devset_results['overall_recall'], 
                                                semi_devset_results['overall_f1'], semi_devset_results['overall_accuracy'],
                                                cf_devset_results['overall_precision'], cf_devset_results['overall_recall'], 
                                                cf_devset_results['overall_f1'], cf_devset_results['overall_accuracy'],
                                                mix_devset_results['overall_precision'], mix_devset_results['overall_recall'], 
                                                mix_devset_results['overall_f1'], mix_devset_results['overall_accuracy']
                                                ]
                        


                    ori_testset_results = eval(ori_trainer, testset)
                    semi_testset_results = eval(semi_trainer, testset)
                    cf_testset_results = eval(cf_trainer, testset)
                    mix_testset_results = eval(mix_trainer, testset)
                    testset_results.loc[seed, :] = [ ori_testset_results['overall_precision'], ori_testset_results['overall_recall'], 
                                            ori_testset_results['overall_f1'], ori_testset_results['overall_accuracy'],
                                            semi_testset_results['overall_precision'], semi_testset_results['overall_recall'], 
                                            semi_testset_results['overall_f1'], semi_testset_results['overall_accuracy'],
                                            cf_testset_results['overall_precision'], cf_testset_results['overall_recall'], 
                                            cf_testset_results['overall_f1'], cf_testset_results['overall_accuracy'],
                                            mix_testset_results['overall_precision'], mix_testset_results['overall_recall'], 
                                            mix_testset_results['overall_f1'], mix_testset_results['overall_accuracy']
                                            ]

                del ori_model, ori_trainer, semi_trainer, cf_trainer, mix_trainer

            devset_results.loc[seeds, :] = devset_results.mean()
            devset_results.to_csv(f'results/{ckpt}-{num_samples}/semi_{semi_aug_ratio}/cf_{cf_aug_ratio}/{eval_task}_dev.csv')
            testset_results.loc[seeds, :] = testset_results.mean()
            testset_results.to_csv(f'results/{ckpt}-{num_samples}/semi_{semi_aug_ratio}/cf_{cf_aug_ratio}/{eval_task}_test.csv')