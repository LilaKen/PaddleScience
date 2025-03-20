# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings

import hydra
from omegaconf import DictConfig
from os import path as osp
import paddle
import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set seed
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.Transolver_Structured_Mesh_2D(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        space_dim=cfg.MODEL.space_dim,
        n_layers=cfg.MODEL.n_layers,
        n_hidden=cfg.MODEL.n_hidden,
        dropout=cfg.MODEL.dropout,
        n_head=cfg.MODEL.n_heads,
        Time_Input=cfg.MODEL.Time_Input,
        mlp_ratio=cfg.MODEL.mlp_ratio,
        fun_dim=cfg.TRAIN.T_in,
        out_dim=cfg.MODEL.out_dim,
        slice_num=cfg.MODEL.slice_num,
        ref=cfg.MODEL.ref,
        unified_pos=cfg.MODEL.unified_pos,
        H=int((64 - 1) / cfg.TRAIN.downsample + 1),
        W=int((64 - 1) / cfg.TRAIN.downsample + 1),
    )

    train_dataloader_cfg = {
        "dataset": {
            "name": "NavierStokesDataset",
            "data_path": cfg.TRAIN.data_path,
            "input_keys": ("input",),
            "label_keys": ("label",),
            "weight_keys": ("weight_keys",),
            "ntrain": cfg.TRAIN.ntrain,
            "ntest": cfg.TRAIN.ntest,
            "T_in": cfg.TRAIN.T_in,
            "T": cfg.TRAIN.T,
            "r": cfg.TRAIN.downsample,
            "h": int((64 - 1) / cfg.TRAIN.downsample + 1),
            "mode": cfg.TRAIN.mode_train,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    navierstokes_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.L2RelLoss("mean"),
        name="navierstokes_constraint",
    )

    constraint = {navierstokes_constraint.name: navierstokes_constraint}

    valid_dataloader_cfg = {
        "dataset": {
            "name": "NavierStokesDataset",
            "data_path": cfg.TRAIN.data_path,
            "input_keys": ("input",),
            "label_keys": ("label",),
            "weight_keys": ("weight_keys",),
            "ntrain": cfg.TRAIN.ntrain,
            "ntest": cfg.TRAIN.ntest,
            "T_in": cfg.TRAIN.T_in,
            "T": cfg.TRAIN.T,
            "r": cfg.TRAIN.downsample,
            "h": int((64 - 1) / cfg.TRAIN.downsample + 1),
            "mode": cfg.TRAIN.mode_eval,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    navierstokes_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.L2RelLoss("mean"),
        metric={"MeanL2Rel": ppsci.metric.MeanL2Rel()},
        name="navierstokes_valid",
    )

    validator = {navierstokes_valid.name: navierstokes_valid}

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.OneCycleLR(
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=(cfg.TRAIN.ntrain // cfg.TRAIN.batch_size + 1),
        max_learning_rate=cfg.TRAIN.lr,
    )()

    optimizer = ppsci.optimizer.AdamW(lr_scheduler, weight_decay=cfg.TRAIN.weight_decay)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=cfg.TRAIN.epochs,
        validator=validator,
        device="cpu",
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad
    )

    # train model
    solver.train()

    solver.eval()


def evaluate(cfg: DictConfig):
    # set seed
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set model
    model = ppsci.arch.Transolver_Structured_Mesh_2D(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        space_dim=cfg.MODEL.space_dim,
        n_layers=cfg.MODEL.n_layers,
        n_hidden=cfg.MODEL.n_hidden,
        dropout=cfg.MODEL.dropout,
        n_head=cfg.MODEL.n_heads,
        Time_Input=cfg.MODEL.Time_Input,
        mlp_ratio=cfg.MODEL.mlp_ratio,
        fun_dim=cfg.TRAIN.T_in,
        out_dim=cfg.MODEL.out_dim,
        slice_num=cfg.MODEL.slice_num,
        ref=cfg.MODEL.ref,
        unified_pos=cfg.MODEL.unified_pos,
        H=int((64 - 1) / cfg.TRAIN.downsample + 1),
        W=int((64 - 1) / cfg.TRAIN.downsample + 1),
    )

    valid_dataloader_cfg = {
        "dataset": {
            "name": "NavierStokesDataset",
            "data_path": cfg.EVAL.data_path,
            "input_keys": ("input",),
            "label_keys": ("label",),
            "weight_keys": ("weight_keys",),
            "ntrain": cfg.EVAL.ntrain,
            "ntest": cfg.EVAL.ntest,
            "T_in": cfg.EVAL.T_in,
            "T": cfg.EVAL.T,
            "r": cfg.EVAL.downsample,
            "h": int((64 - 1) / cfg.EVAL.downsample + 1),
            "mode": cfg.mode,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "num_workers": cfg.EVAL.num_workers,
    }

    navierstokes_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.L2RelLoss("mean"),
        metric={"MeanL2Rel": ppsci.metric.MeanL2Rel()},
        name="navierstokes_valid",
    )

    validator = {navierstokes_valid.name: navierstokes_valid}

    solver = ppsci.solver.Solver(
        model=model,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad
    )

    # evaluate model
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="Transolver_PDE_StandardBenchmark_NS.yaml")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore")
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
