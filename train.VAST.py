# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
from sklearn.metrics import f1_score
import json

import random
import sys
import codecs
import numpy as np
import pandas as pd
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

# from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer as RobertaTokenizer
from transformers.optimization import AdamW
# from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.bert.modeling_bert import BertModel as RobertaModel

# from transformers.modeling_bert import BertModel
# from transformers.tokenization_bert import BertTokenizer
# from bert_common_functions import store_transformers_models

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 768 #1024
MLP_hidden_dim = 283
pretrain_model_dir = 'bert-base-uncased' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'

roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)

def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print('starting model storing....')
    # model.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir)
    # tokenizer.save_pretrained(output_dir)
    print('store succeed')

class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        # self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    # def forward(self, input_ids, input_mask):
    def forward(self, outputs_single):
        # outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single



class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, MLP_hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(MLP_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class StdProcessor():
    """Processor for the Stance Detection datasets (GLUE version)."""

#######################################################################

    def get_SemT6_test(self):
        print('\nGetting SemT6 test set\n')
        dic = {'AGAINST':'against', 'FAVOR':'support', 'NONE':'neutral'}

        test_examples = []

        df_1 = pd.read_csv('dataset/SemT6/trainingdata-all-annotations.txt', sep="	")
        df_2 = pd.read_csv('dataset/SemT6/testdata-taskA-all-annotations.txt', sep="	")
        df_3 = pd.read_csv('dataset/SemT6/testdata-taskB-all-annotations.txt', sep="	")
        df_12 = df_1.append(df_2, ignore_index=True)
        df_test = df_12.append(df_3, ignore_index=True)

        df_test = df_test[['Tweet','Target','Stance']]
        df_test['Stance'] = [dic[i] for i in df_test['Stance']]

        for i in range(len(df_test)):
            guid = "test-"+str(i) #change train to test
            text_a = df_test['Target'][i]
            text_b = df_test['Tweet'][i]
            label = df_test['Stance'][i]

            test_examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return test_examples


    def get_VAST_new(self, filelists):

        dic = {0:'against',1:'support',2:'neutral'}

        data_list = []
        for fil in filelists:
            examples = []
            fin = open(fil, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            for i in range(0, len(lines), 4):
                premise = lines[i].lower().strip()
                hypothesis = lines[i+1].lower().strip()
                label = dic[int(lines[i+2].strip())]


                examples.append(
                                InputExample(guid=guid, text_a=premise, text_b=hypothesis, label=label))
            data_list.append(examples)


        return data_list[0], data_list[1], data_list[2]


    def get_VAST(self, filelists):

        dic = {0:'against',1:'support',2:'neutral'}

        data_list = []
        for fil in filelists:
            examples = []
            df_train = pd.read_csv(fil)
            df_train = df_train[['text','topic','label']]
            df_train['label'] = [dic[i] for i in df_train['label']]
            for i in range(len(df_train)):
                guid = "train-"+str(i)

                raw_premise = json.loads(df_train['text'][i])
                premise = ''
                for lis in raw_premise:
                    premise+=' '+' '.join(lis)
                premise = premise.strip()

                raw_hypothesis = json.loads(df_train['topic'][i])
                hypothesis = ' '.join(raw_hypothesis)

                label = df_train['label'][i]

                # print("premise -->:", premise)
                # print("hypothesis -->:", hypothesis)

                examples.append(
                                InputExample(guid=guid, text_a=premise, text_b=hypothesis, label=label))
            data_list.append(examples)


        return data_list[0], data_list[1], data_list[2]



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 sep_token_extra=True,###################
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # label_map = {'against':0,'support':1,'neutral':2} ##################
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 2000 == 0:################################
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        #  BERT:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1

        # RoBERTa:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0     0    1  1  1  1  1   1
        #  segment_id

        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features #a list of InputFeatures objects


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()




def examples_to_dataloader(examples, label_list, max_seq_length, tokenizer, output_mode, batch_size, sample_type='random'):

    dev_features = convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    dev_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    dev_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    dev_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    dev_all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)

    dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids)
    if sample_type == 'random':
        dev_sampler = RandomSampler(dev_data)
    else:
        dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)
    return dev_dataloader





def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

#     parser.add_argument('--pretrain_epochs',
#                         type=int,
#                         default=5,
#                         help="random seed for initialization")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
#     parser.add_argument("--pretrain_batch_size",
#                         default=16,
#                         type=int,
#                         help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
#     parser.add_argument('--pretrain_sample_size',
#                         type=int,
#                         default=50,
#                         help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


    args = parser.parse_args()

    processors = {
        "rte": StdProcessor

    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

#     if not args.do_train and not args.do_eval:
#         raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))


    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    data_path = '/home/tup51337/dataset/VAST_Bin_Liang/'
    train_examples, dev_examples, test_examples = processor.get_VAST_new([data_path+'lda_vast_train.raw', data_path+'vast_dev.raw', data_path+'vast_test.raw'])

    label_list = ["against", "support", "neutral"]
    num_labels = len(label_list)

    print('num_labels:', num_labels, 'training size:', len(train_examples), 'dev size:', len(dev_examples), 'test size:', len(test_examples))


    num_train_optimization_steps = None
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    model = RobertaForSequenceClassification(num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.to(device)
    roberta_single.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_dev_F1 = 0.0
    final_test_f1 = 0.0
    tr_loss_track = []
    dev_loss_track = []
    if args.do_train:
        train_dataloader = examples_to_dataloader(train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args.train_batch_size, sample_type='random')
        dev_dataloader = examples_to_dataloader(dev_examples, label_list, args.max_seq_length, tokenizer, output_mode, args.eval_batch_size, sample_type='sequential')
        test_dataloader = examples_to_dataloader(test_examples, label_list, args.max_seq_length, tokenizer, output_mode, args.eval_batch_size, sample_type='sequential')

        '''train on training data'''
        iter_co = 0
        final_test_performance = 0.0
        for fine_tune_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # logits = model(input_ids, input_mask)
                outputs_single = roberta_single(input_ids, input_mask, None)
                logits = model(outputs_single)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1)) ##

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            '''
            start evaluate on dev set after each epoch
            '''
            model.eval()
            preds = []
            gold_label_ids = []
            dev_loss = 0
            loss_fct = nn.CrossEntropyLoss()
            # print('Evaluating...')
            for input_ids, input_mask, segment_ids, label_ids in dev_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                gold_label_ids+=list(label_ids.detach().cpu().numpy())

                with torch.no_grad():
                    # logits = model(input_ids, input_mask)
                    outputs_single = roberta_single(input_ids, input_mask, None)
                    logits = model(outputs_single)
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
            #####
                dev_loss += loss_fct(logits.view(-1, num_labels),label_ids.view(-1)).item()

            print('\ntrain loss(per sample):', str(tr_loss/len(train_examples)), 'dev loss(per sample):', str(dev_loss/len(dev_examples)),'\n')
            #####

            preds = preds[0]
            pred_probs = softmax(preds,axis=1)#變sum=1的小數
            pred_label_ids = list(np.argmax(pred_probs, axis=1))
            gold_label_ids = gold_label_ids

            assert len(pred_label_ids) == len(gold_label_ids)


            hit_co = 0
            for k in range(len(pred_label_ids)):
                if pred_label_ids[k] == gold_label_ids[k]:
                    hit_co +=1
            dev_acc = hit_co/len(gold_label_ids)
            dev_F1 = f1_score(gold_label_ids, pred_label_ids, average='macro')

##########################################################
            # use F1 to select best model
            if dev_F1 > max_dev_F1:
                max_dev_F1 = dev_F1
                print('\ndev F1:', dev_F1, 'max dev F1:', max_dev_F1, ' dev acc:', dev_acc, '\n')
                '''Test this best model on test set'''

                model.eval()
                preds = []
                gold_label_ids = []
                for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    gold_label_ids+=list(label_ids.detach().cpu().numpy())

                    with torch.no_grad():
                        # logits = model(input_ids, input_mask)
                        outputs_single = roberta_single(input_ids, input_mask, None)
                        logits = model(outputs_single)
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                preds = preds[0]
                pred_probs = softmax(preds,axis=1)#變sum=1的小數
                pred_label_ids = list(np.argmax(pred_probs, axis=1))
                gold_label_ids = gold_label_ids

                assert len(pred_label_ids) == len(gold_label_ids)


                hit_co = 0
                for k in range(len(pred_label_ids)):
                    if pred_label_ids[k] == gold_label_ids[k]:
                        hit_co +=1
                test_acc = hit_co/len(gold_label_ids)
                test_F1 = f1_score(gold_label_ids, pred_label_ids, average='macro')
                final_test_f1 = test_F1
                print('\ncurrent test F1:', test_F1)

            else:
                print('\ndev F1:', dev_F1, 'max dev F1:', max_dev_F1, ' dev acc:', dev_acc, '\n')

        print('Final test f1: ', final_test_f1)

if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=3 python -u train.VAST.py --task_name rte --do_train --do_lower_case --num_train_epochs 20 --train_batch_size 64 --eval_batch_size 32 --learning_rate 1e-3 --max_seq_length 205 --seed 42


'''
