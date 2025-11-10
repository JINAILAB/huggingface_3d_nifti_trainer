#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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



import logging
import sys
import os
import json
from dataclasses import dataclass, field

from datasets import load_from_disk, load_dataset
import transformers
from transformers import (
                          AutoConfig, 
                          AutoModelForImageClassification, 
                          TrainingArguments, 
                          Trainer, 
                          EarlyStoppingCallback, 
                          set_seed,
                          HfArgumentParser
                          )
import data_presets
import torch
import datetime
from image_utils import save_confusion_matrix, compute_metrics, create_hook, get_embedding_layer, save_umap, check_embedding

import numpy as np
import wandb

from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class CustomTrainingArguments:
    """
    Custom training arguments that are not included in the standard TrainingArguments.
    """
    early_stopping_epoch: int = field(
        default=0,
        metadata={"help": "Number of epochs with no improvement after which training will be stopped. Set to 0 to disable early stopping."}
    )
    umap: bool = field(
        default=False,
        metadata={"help": "Whether to generate UMAP visualization after training."}
    )
    gradcam: bool = field(
        default=False,
        metadata={"help": "Whether to generate GradCAM visualization after training."}
    )
    project_name: str = field(
        default="huggingface_3d_nifti",
        metadata={"help": "Project name for wandb and output directory."}
    )
    
    # Image size parameters
    crop_size: tuple = field(
        default=(64, 64, 64),
        metadata={"help": "Crop size for 3D images (depth, height, width)."}
    )
    
    # Normalization parameters
    clip_min_max: tuple = field(
        default=None,
        metadata={"help": "Intensity clipping range (min, max). If provided, clips intensity values before normalization. Example: (-1000, 1000) for CT scans."}
    )
    use_normalize_intensity: bool = field(
        default=True,
        metadata={"help": "If True, use NormalizeIntensity (subtract mean, divide by std). If False, use ScaleIntensity (linear scaling to [scale_minv, scale_maxv])."}
    )
    normalize_nonzero: bool = field(
        default=True,
        metadata={"help": "For NormalizeIntensity: if True, only normalize non-zero values. Useful for images with background zeros."}
    )
    scale_minv: float = field(
        default=0.0,
        metadata={"help": "For ScaleIntensity: minimum value of output range."}
    )
    scale_maxv: float = field(
        default=1.0,
        metadata={"help": "For ScaleIntensity: maximum value of output range."}
    )
    
    # Intensity augmentation parameters
    gaussian_noise_prob: float = field(
        default=0.15,
        metadata={"help": "Probability of applying Gaussian noise augmentation."}
    )
    gaussian_noise_std: float = field(
        default=0.1,
        metadata={"help": "Standard deviation for Gaussian noise."}
    )
    gaussian_smooth_prob: float = field(
        default=0.15,
        metadata={"help": "Probability of applying Gaussian smoothing."}
    )
    gaussian_smooth_sigma: tuple = field(
        default=(0.5, 1.0),
        metadata={"help": "Sigma range for Gaussian smoothing."}
    )
    shift_intensity_prob: float = field(
        default=0.15,
        metadata={"help": "Probability of applying intensity shift."}
    )
    shift_intensity_offset: float = field(
        default=0.1,
        metadata={"help": "Offset value for intensity shift."}
    )
    scale_intensity_prob: float = field(
        default=0.15,
        metadata={"help": "Probability of applying intensity scaling."}
    )
    scale_intensity_factors: tuple = field(
        default=(0.9, 1.1),
        metadata={"help": "Scaling factor range for intensity scaling."}
    )
    adjust_contrast_prob: float = field(
        default=0.15,
        metadata={"help": "Probability of applying contrast adjustment."}
    )
    contrast_range: tuple = field(
        default=(0.75, 1.25),
        metadata={"help": "Gamma range for contrast adjustment."}
    )
    gaussian_sharpen_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of applying Gaussian sharpening."}
    )
    sharpen_sigma: tuple = field(
        default=(0.5, 1.0),
        metadata={"help": "Sigma range for Gaussian sharpening."}
    )
    histogram_shift_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of applying histogram shift."}
    )
    
    # Geometric augmentation parameters
    flip_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of applying random flips on each axis."}
    )
    rotate_prob: float = field(
        default=0.2,
        metadata={"help": "Probability of applying random rotation."}
    )
    rotate_range: float = field(
        default=0.174,
        metadata={"help": "Rotation range in radians (default: ±10 degrees)."}
    )
    zoom_prob: float = field(
        default=0.2,
        metadata={"help": "Probability of applying random zoom."}
    )
    zoom_range: tuple = field(
        default=(0.9, 1.1),
        metadata={"help": "Zoom factor range (default: ±10%)."}
    )
    affine_prob: float = field(
        default=0.2,
        metadata={"help": "Probability of applying affine transformation."}
    )
    affine_rotate_range: float = field(
        default=0.174,
        metadata={"help": "Affine rotation range in radians."}
    )
    affine_shear_range: float = field(
        default=0.1,
        metadata={"help": "Affine shear range (default: ±10%)."}
    )
    affine_scale_range: tuple = field(
        default=(0.9, 1.1),
        metadata={"help": "Affine scale range (default: ±10%)."}
    )
    grid_distortion_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of applying grid distortion."}
    )
    grid_distortion_num_cells: int = field(
        default=5,
        metadata={"help": "Number of cells for grid distortion."}
    )
    grid_distortion_distort_limit: float = field(
        default=0.05,
        metadata={"help": "Distortion limit for grid distortion."}
    )
    
    # Simulated low resolution parameters
    coarse_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of applying coarse dropout (simulated low resolution)."}
    )
    coarse_dropout_holes: int = field(
        default=8,
        metadata={"help": "Number of holes for coarse dropout."}
    )
    coarse_dropout_spatial_size: tuple = field(
        default=(4, 4, 4),
        metadata={"help": "Spatial size for coarse dropout holes."}
    )
    
    # GPU and DataLoader parameters
    gpu_ids: Optional[str] = field(
        default=None,
        metadata={"help": "GPU IDs to use (e.g., '0,1' or '0'). If not specified, uses default GPU selection."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """
    local_dataset : bool = field(default=True, metadata={"help": "Use local dataset. If False, use dataset from the hub."})
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub or the path to a local dataset"
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    nifti_column_name: str = field(
        default="nifti",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )
    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError(
                "You must specify a dataset name from the hub or a local dataset path"
            )



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="nwirandx/medicalnet-resnet3d10",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    use_pretrained: bool = field(
        default=True,
        metadata={"help": "If True, use pretrained model from huggingface."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    
def setup_logging(logger,training_args):
    # Setup logging - force=True to override any existing configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
        force=True
    )

    # Set transformers logging verbosity
    transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
        
    setup_logging(logger, training_args)
    
    # Set GPU devices if specified
    if custom_args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = custom_args.gpu_ids
        logger.info(f"Using GPU(s): {custom_args.gpu_ids}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # set output directory
    now = datetime.datetime.now()
    now = now.strftime("%Y%m%d_%H%M")

    model_name = model_args.model_name_or_path.split("/")[-1]
    output_dir = os.path.join('runs', custom_args.project_name, model_name + '_' + now)
    
    # initialize wandb
    if 'wandb' in training_args.report_to:
        wandb.init(project=custom_args.project_name, name=model_name + '_' + now, dir=output_dir)
        wandb.config.update({'dataset_name' : data_args.dataset_name})
        wandb.config.update(custom_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
    # Save config to JSON
    os.makedirs(output_dir, exist_ok=True)
    config_dict = {**vars(model_args), **vars(data_args), **vars(training_args), **vars(custom_args)}
    with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)

    dataset = load_from_disk(data_args.dataset_name) if data_args.local_dataset else load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    
    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names

    if data_args.nifti_column_name not in dataset_column_names:
        raise ValueError(
            f"--nifti_column_name {data_args.nifti_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--nifti_column_name` to the correct nifti column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )
        
    labels = dataset["train"].features[data_args.label_column_name].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    
    train_ds, valid_ds, train_transform, valid_transform = data_presets.load_dataset(dataset, custom_args)
    
    # Load model based on use_pretrained flag
    if model_args.use_pretrained:
        logger.info("Loading pretrained model from %s", model_args.model_name_or_path)
        model = AutoModelForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True, 
            trust_remote_code=model_args.trust_remote_code,
            # cache_dir=tempfile.mkdtemp() # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
    else:
        logger.info("Creating model from config (randomly initialized) based on %s", model_args.model_name_or_path)
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            trust_remote_code=model_args.trust_remote_code,
        )
        model = AutoModelForImageClassification.from_config(config, trust_remote_code=model_args.trust_remote_code)

    training_args.logging_dir = output_dir
    training_args.output_dir = output_dir 
    
    trainer_callbacks = [EarlyStoppingCallback(early_stopping_patience=custom_args.early_stopping_epoch)] if custom_args.early_stopping_epoch > 0 else None
    
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x[data_args.label_column_name] for x in batch])
        }
        
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        # tokenizer=train_transform,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=trainer_callbacks
    )
    

    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best.hf'))
    trainer.save_state()

    
    if custom_args.umap:
        hook, embedding_outputs = create_hook()
        embedding = get_embedding_layer(model)
        hook_handle = embedding.register_forward_hook(hook)
    
    preds_output= trainer.predict(valid_ds)
    
    y_preds = np.argmax(preds_output.predictions, axis=-1)
    y_valid = np.array(valid_ds['label'])
    save_confusion_matrix(y_preds, y_valid, labels, output_dir)
    
    if custom_args.umap:
        all_embeddings = check_embedding(model, embedding_outputs)
        save_umap(all_embeddings, y_valid, labels, output_dir)
        hook_handle.remove()
    
    # if custom_args.gradcam:
    #     save_gradcam(model, id2label, output_dir, valid_ds, valid_transform, custom_args.crop_size)
    
    if training_args.report_to == 'wandb':
        wandb.finish(exit_code=0) 
    
    
if __name__ == '__main__':
    main()