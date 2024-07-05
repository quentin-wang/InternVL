import logging
import math
import os
import random
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

import orjson as json
import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator, pad_data_collator,
                            replace_llama2_attn_with_flash_attn,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess,
                                    find_closest_aspect_ratio, preprocess,
                                    preprocess_internlm, preprocess_mpt)
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, LlamaConfig, LlamaForCausalLM,
                          LlamaTokenizer, Trainer, TrainingArguments,
                          default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

from internvl.train.dpo_trainer import DPOTrainer

# Upgrade transformers to v4.36.2, we don't need it anymore
# replace_llama2_attn_with_flash_attn()
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()  # Can only automatically infer lengths for datasets whose items are dictionaries with an 'input_ids' key.

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v1',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )

    # =============================================================
    # only_lora_ffn: bool = True
    moe_enable: bool = False

    # =============================================================

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internvl_zh', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=6,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )

    dpo_beta: float = 0.1
    dpo_token_weight: float = 3.0  # 5.0 in the RLHF-V paper for fine-grained DPO


@dataclass
class DPOTrainingArguments(TrainingArguments):
    task: str = field(
        default='DPO',
        metadata={
            'help': 'SFT for language modeling. DPO for direct preference optimization'
        }
    )
    dpo_use_average: bool = True
    dpo_token_weighted: bool = False


from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Sequence, Tuple, Union)
from functools import partial, wraps
from queue import Empty, Queue
import multiprocess
from tqdm import tqdm
import numpy as np


class LLMDataset(Dataset):

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        if isinstance(idx, int):
            # data, _ = self.data[idx]
            data = self.data[idx]
            return data
        elif isinstance(idx, str):
            return [d[0][idx] for d in self.data]
        else:
            raise ValueError(f'idx: {idx}')

    def select(self, idx_list: List[int]) -> 'LLMDataset':
        data = [self.data[i] for i in idx_list]
        return self.__class__(data)

    def __len__(self) -> int:
        return len(self.data)


MapFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


def _single_map(d: Dict[str, Any],
                map_func: MapFunc) -> Optional[Dict[str, Any]]:
    d = map_func(d)
    if len(d) == 0:
        return None
    return d


def _map_mp_single(subset, map_func: MapFunc, queue: Queue,
                   start_idx: int):
    for i, d in enumerate(subset, start=start_idx):
        queue.put((i, map_func(d)))  # idx, result


def _map_mp_i(dataset, map_func: MapFunc,
              num_proc: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with multiprocess.Pool(
            num_proc) as pool, multiprocess.Manager() as manager:
        queue = manager.Queue()
        async_results = []
        split_idx = np.linspace(0, len(dataset), num_proc + 1, dtype=np.int32)
        for i in range(num_proc):
            subset = dataset.select(range(split_idx[i], split_idx[i + 1]))
            async_results.append(
                pool.apply_async(
                    _map_mp_single,
                    args=(subset, map_func, queue, split_idx[i])))
        while True:
            try:
                yield queue.get(timeout=0.05)
            except Empty:
                if all(async_result.ready()
                       for async_result in async_results) and queue.empty():
                    break


def _map_mp(dataset, map_func: MapFunc,
            num_proc: int) -> List[Dict[str, Any]]:
    # Solving the unordered problem
    data = [None] * len(dataset)
    num_proc = min(num_proc, len(dataset))
    for d in tqdm(_map_mp_i(dataset, map_func, num_proc), total=len(dataset)):
        data[d[0]] = d[1]
    return data


def dataset_map(dataset,
                map_func: MapFunc,
                num_proc: int = 1) -> Optional[LLMDataset]:
    single_map = partial(_single_map, map_func=map_func)
    if num_proc == 1:
        data = []
        for d in tqdm(dataset):
            d = single_map(d)
            data.append(d)
    else:
        assert num_proc > 1
        data = _map_mp(dataset, single_map, num_proc)
    data = [d for d in data if d is not None]
    if len(data) == 0:
        logger.warning('len(dataset): 0')
        return None
    return LLMDataset(data)


def convert_format(data_item):
    if "query" in data_item:
        image_value = data_item["image"]
        query_value = data_item["query"]
        response_value = str(data_item["response"])  # 确保响应是字符串格式

        conversations = [
            {"from": "human", "value": f"<image>\n{query_value}"},
            {"from": "gpt", "value": response_value}
        ]

        new_data_item = {
            "image": image_value,
            "conversations": conversations
        }
        return new_data_item

    return data_item


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, tcs_loader, num_image_token,
                 image_size=224, is_train=True, pad2square=False, group_by_length=False,
                 dynamic_image_size=False, use_thumbnail=False, min_dynamic_patch=1,
                 max_dynamic_patch=6, repeat_time=1, normalize_type='imagenet'):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # choice top len(self.raw_data) * repeat_time samples
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        if self.group_by_length:
            self.conv2length = {}  # using dict to speedup the calculation of token length
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                # data_item = self.convert_format(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # use precomputed length if exists
                else:
                    # compute token length using tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def multi_modal_get_item(self, data_item):
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # fix bug when there are multiple <image> in conversations
        image_cnt = 0
        for idx, conv in enumerate(data_item['conversations']):
            conv['value'] = conv['value'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
            if idx == 0:
                conv['value'] = '<image>\n' + conv['value']
            image_cnt += conv['value'].count('<image>')
        assert image_cnt == 1, f'There should be exactly one <image> in the conversation, but got {image_cnt}'

        if data_item['image'].startswith('s3://'):
            image_path = self.root + data_item['image']
        else:
            image_path = os.path.join(self.root, data_item['image'])
        if self.tcs_loader is not None:
            image = self.tcs_loader(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        else:
            preprocess_function = preprocess
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * num_patches,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        else:
            preprocess_function = preprocess
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * num_patches, text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # i = i % len(self.raw_data) # or will run into infinite loop.
        if i >= self.__len__():
            raise StopIteration

        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and len(data_item['image']) != 0:
                    ret = self.multi_modal_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                print(e)
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if data_item['image'].startswith('s3://'):
                        data_path = self.root + data_item['image']
                    else:
                        data_path = os.path.join(self.root, data_item['image'])
                    print(f'Failed to load image: exception: {e}')
                    print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(data_args, tokenizer, tcs_loader, model, group_by_length=False,
                   dynamic_image_size=False, use_thumbnail=False, min_dynamic_patch=1,
                   max_dynamic_patch=6, normalize_type='imagenet'):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_name in ds_collections.keys():
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_dynamic_patch = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_dynamic_patch} according to the meta file')
        try:
            dataset = LazySupervisedDataset(
                data_args.conv_style, ds_collections[ds_name],
                tokenizer,
                tcs_loader,
                num_image_token=model.num_image_token,
                image_size=data_args.force_image_size,
                is_train=ds_collections[ds_name]['data_augment'],
                pad2square=data_args.pad2square,
                group_by_length=group_by_length,
                dynamic_image_size=dynamic_image_size,
                use_thumbnail=use_thumbnail,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                repeat_time=repeat_time,
                normalize_type=normalize_type,
            )
            dataset.ds_name = ds_name
            dataset = dataset_map(dataset=dataset, map_func=convert_format)

        except Exception as e:
            logger.info(f'Error in loading dataset: {ds_name}, e: {e}')
            exit()
        
        repeat_time = 1 if repeat_time < 1 else repeat_time  # don't repeat if repeat_time is less than 1
        for i in range(repeat_time):
            logger.info(f'Add dataset:{ds_name}_{i} with length: {len(dataset)}')
            datasets.append(dataset)
            if data_args.use_data_resampling:
                lengths.append(math.sqrt(len(dataset)))
            else:
                lengths.append(len(dataset))
    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


# DPO

# The only other function needed
def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out


# This is the collator function for the preference during training 
def preference_collator_fn(instances, pad_token_id):
    rej_instances, win_instances = list(zip(*instances))
    concat_batch = concat_pad_data_collator([*win_instances, *rej_instances], pad_token_id)  # win first.
    rej_batch = concat_pad_data_collator(rej_instances, pad_token_id)  # TODO: needded to verify
    win_batch = concat_pad_data_collator(win_instances, pad_token_id)
 
    # concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    # concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], -100)
    # concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

    batch = dict(
        concatenated_input_ids=concat_batch['input_ids'],
        concatenated_labels=concat_batch['labels'],
        concatenated_attention_mask=concat_batch['attention_mask'],
        concatenated_pixel_values=concat_batch['pixel_values'],
        concatenated_image_flags=concat_batch['image_flags'],

        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],
        win_attention_mask=win_batch['attention_mask'],
        rej_attention_mask=rej_batch['attention_mask'],
        # images=win_batch['images'],
    )
    # print(f"+++11 batch['concatenated_image_flags'] ", concat_batch['image_flags'], "labels", concat_batch['labels'])
    return batch


@dataclass
class DataCollatorForDPODataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    beta: float
    mod_token_weight: float

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = preference_collator_fn(instances, self.tokenizer.pad_token_id)

        rej_instances, win_instances = list(zip(*instances))

        batch['beta'] = self.beta

        # batch['ref_win_logp'] = torch.as_tensor([x['ref_win_logp'] for x in win_instances])
        # batch['ref_rej_logp'] = torch.as_tensor([x['ref_rej_logp'] for x in rej_instances])
        # batch['ref_win_avg_logp'] = torch.as_tensor([x['ref_win_avg_logp'] for x in win_instances])
        # batch['ref_rej_avg_logp'] = torch.as_tensor([x['ref_rej_avg_logp'] for x in rej_instances])

        # ref_win_per_token_logp = [torch.as_tensor(x['ref_win_per_token_logp']) for x in win_instances]
        # ref_rej_per_token_logp = [torch.as_tensor(x['ref_rej_per_token_logp']) for x in rej_instances]

        # batch['ref_win_per_token_logp'] = torch.nn.utils.rnn.pad_sequence(ref_win_per_token_logp, batch_first=True, padding_value=0)
        # batch['ref_rej_per_token_logp'] = torch.nn.utils.rnn.pad_sequence(ref_rej_per_token_logp, batch_first=True, padding_value=0)

        # win_input_ids = batch['win_input_ids']
        # rej_input_ids = batch['rej_input_ids']
        # win_labels = batch['win_labels']
        # rej_labels = batch['rej_labels']
        # # assert batch['ref_win_per_token_logp'].size(1) >= win_input_ids.size(1) - 1, f"{batch['ref_win_per_token_logp'].size(1)} >= {win_input_ids.size(1) - 1}"
        # # assert batch['ref_rej_per_token_logp'].size(1) >= rej_input_ids.size(1) - 1, f"{batch['ref_rej_per_token_logp'].size(1)} >= {rej_input_ids.size(1) - 1}"

        # # length of logp is one-token shorter since the last token's output is not used
        # batch['ref_win_per_token_logp'] = batch['ref_win_per_token_logp'][:, :win_input_ids.size(1) - 1]
        # batch['ref_rej_per_token_logp'] = batch['ref_rej_per_token_logp'][:, :rej_input_ids.size(1) - 1]
        
        # ========================================

        # The following line 876 to 895 is not used as we are not doing fine-grained DPO
        # win_token_weight = torch.ones_like(batch['ref_win_per_token_logp'])
        # rej_token_weight = torch.ones_like(batch['ref_rej_per_token_logp'])

        # for idx, (w, r, wl, rl, wlogp, rlogp) in enumerate(zip(win_input_ids, rej_input_ids, win_labels, rej_labels, ref_win_per_token_logp, ref_rej_per_token_logp)):
        #     valid_w = w[1:]
        #     valid_r = r[1:]

        #     min_match_size = 3
        #     # TODO: add junk condition for space tokens like 13 for '\n'
        #     r_mod, w_mod = get_diff_ids(valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
        #     r_mod_tokens = valid_r[r_mod]
        #     w_mod_tokens = valid_w[w_mod]

        #     win_token_weight[idx][w_mod] = self.mod_token_weight
        #     rej_token_weight[idx][r_mod] = self.mod_token_weight

        # batch['win_token_weight'] = win_token_weight
        # batch['rej_token_weight'] = rej_token_weight
        # batch['concatenated_token_weight'] = concate_pad(win_token_weight, rej_token_weight, 0)

        for ins in win_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
        for ins in rej_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
        return batch
    

class DPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, datasets):
        """
        datasets is a list of dataset which consist of chosen, reject and may be general.
        """

        super(DPODataset, self).__init__()
        self.raw_data = datasets

    def __len__(self):
        return len(self.raw_data['win'])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data['win'])
        result = []
        subsets = [self.raw_data['rej'], self.raw_data['win']]
        for d in subsets:
            result.append(d[i])
        return result
        # return rej_data_dict, win_data_dict

        # while True:
        #     try:
        #         for d in subsets:
        #             data_item = json.loads(d[i])
        #             if 'image' in data_item and len(data_item['image']) != 0:
        #                 ret = d.multi_modal_get_item(data_item)
        #             else:
        #                 ret = d.pure_text_get_item(data_item)
        #             result.append(ret)
        #         break
        #     except Exception as e:
        #         print(e)
        #         data_item = json.loads(self.raw_data[i])
        #         if 'image' in data_item:
        #             if data_item['image'].startswith('s3://'):
        #                 data_path = self.root + data_item['image']
        #             else:
        #                 data_path = os.path.join(self.root, data_item['image'])
        #             print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
        #         i = random.randint(0, len(self.raw_data) - 1)


def build_dpo_datasets(data_args, tokenizer, tcs_loader, model, group_by_length=False,
                   dynamic_image_size=False, use_thumbnail=False, min_dynamic_patch=1,
                   max_dynamic_patch=6, normalize_type='imagenet'):
    # pad_sequence stacks a list of Tensors along a new dimension, and pads them to equal length. 

    datasets = {}
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_name in ds_collections.keys():
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_dynamic_patch = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_dynamic_patch} according to the meta file')
        try:
            dataset = LazySupervisedDataset(
                data_args.conv_style, ds_collections[ds_name],
                tokenizer,
                tcs_loader,
                num_image_token=model.num_image_token,
                image_size=data_args.force_image_size,
                is_train=ds_collections[ds_name]['data_augment'],
                pad2square=data_args.pad2square,
                group_by_length=group_by_length,
                dynamic_image_size=dynamic_image_size,
                use_thumbnail=use_thumbnail,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                repeat_time=repeat_time,
                normalize_type=normalize_type,
            )
            dataset.ds_name = ds_name
            dataset = dataset_map(dataset=dataset, map_func=convert_format)

        except Exception:
            logger.info(f'Error in loading dataset: {ds_name}')
            exit()
        
        repeat_time = 1 if repeat_time < 1 else repeat_time  # don't repeat if repeat_time is less than 1
        for i in range(repeat_time):
            logger.info(f'Add dataset:{ds_name}_{i} with length: {len(dataset)}')
            datasets[ds_name] = dataset
            if data_args.use_data_resampling:
                lengths.append(math.sqrt(len(dataset)))
            else:
                lengths.append(len(dataset))

        print(f"+++ load {ds_name}")

    dpo_dataset = DPODataset(datasets)
    return dpo_dataset


def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DPOTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        # config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        # config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA  # deleted
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        # model = InternVLChatModel.from_pretrained(
        #     model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
        llm = AutoModelForCausalLM.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id
    model.neftune_alpha = data_args.neftune_alpha

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_params(module):
        for param in module.parameters():
            param.requires_grad = True

    def _unfreeze_named_params(module, string):
        for name,param in module.named_parameters():
            if string in name:
                param.requires_grad = True

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        # model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    if model_args.moe_enable:
        for layer_idx, layer in enumerate(model.language_model.model.layers):
            if (config.llm_config.n_routed_experts is not None and \
                layer_idx >= config.llm_config.first_k_dense_replace and layer_idx % config.llm_config.moe_layer_freq == 0):
                    _unfreeze_params(layer.feed_forward)

    _unfreeze_named_params(model.language_model.model, "22.feed_forward.w2")
    model.language_model.enable_input_require_grads()
    print(f"+++ model {model}")

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()

    if training_args.task == "SFT":
        train_dataset = build_datasets(
            data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
            dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
            normalize_type=data_args.normalize_type)

        # do we need default_data_collator?
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=concat_pad_data_collator
        )
    elif training_args.task == "DPO":
        train_dataset = build_dpo_datasets(
            data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
            dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
            normalize_type=data_args.normalize_type)

        dpo_data_collator = DataCollatorForDPODataset(tokenizer=tokenizer, beta=data_args.dpo_beta, mod_token_weight=data_args.dpo_token_weight)
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=dpo_data_collator
        )

    else:
        raise Exception(f"Non exists task {training_args.task}")

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
