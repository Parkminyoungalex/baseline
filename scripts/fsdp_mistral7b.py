from typing import Callable, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig
import torch

import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.gpt.model import GPTConfig7B, MistralConfig7B, GPTModel, MistralModel
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
)
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback
from scripts.performance.helpers import set_primary_perf_configs, set_mcore_fsdp_configs
from scripts.performance.utils import get_comm_overlap_callback_idx

def model() -> run.Config[pl.LightningModule]:
    config = run.Config(
        MistralConfig7B,
        seq_length=8192,
        gradient_accumulation_fusion=True,
        init_model_with_meta_device=True,
        use_transformer_engine_full_layer_spec=False,
        share_embeddings_and_output_weights=True,
        deallocate_pipeline_outputs=False,
    )
    return run.Config(MistralModel, config=config)

def trainer(
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 2,
    pipeline_parallelism_type: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = None,
    sequence_parallelism: bool = True,
    num_nodes: int = 64,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism, # attention에서 activation을 sequence 방향으로 
        sequence_parallel=sequence_parallelism, # tp랑 같이 쓰이는데 tp 키면 자동으로 켜져야하지않을까
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim"
        ),
        fsdp="megatron",
        progress_interval=5,
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        val_check_interval=2000,
    )

    return trainer

def pretrain_recipe(
    global_batch_size=4,
    micro_batch_size=1,
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 4,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    num_nodes: int = 1,
    num_gpus_per_node: int = 4,
    performance_mode: bool = True,
    max_steps: int = 100,
    fn: Callable = pretrain,
) -> run.Partial:
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            tensor_parallelism=tensor_parallelism,
            pipeline_parallelism=pipeline_parallelism,
            pipeline_parallelism_type=torch.bfloat16,
            virtual_pipeline_parallelism=None,
            context_parallelism=context_parallelism,
            sequence_parallelism=sequence_parallelism,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            max_steps=max_steps,
            callbacks=[run.Config(TimingCallback, log_tokens_per_sec=True)],
        ),
        data=run.Config(
            PreTrainingDataModule,
            paths=["/root/nemo/dataset/openwebtext"],
            seq_length=8192,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=0.9e-4),
    )

    if performance_mode:
        recipe = pretrain_performance_optimizations(recipe)

    recipe = set_primary_perf_configs(
        recipe,
        task="pretrain",
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        mbs=micro_batch_size,
        gbs=global_batch_size,
        max_steps=max_steps,
        tp_size=tensor_parallelism,
        pp_size=pipeline_parallelism,
        cp_size=context_parallelism,
        vp_size=None,
        ep_size=1,
        use_fsdp_double_buffer=True, # (check)
        use_mcore_fsdp=True,
    )

    return recipe

def pretrain_performance_optimizations(recipe: run.Partial) -> run.Partial:
    if not recipe.trainer.callbacks:
        recipe.trainer.callbacks = []

    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=100,
    )
    mcomm_overlap_callback = run.Config(
        MegatronCommOverlapCallback,
        tp_comm_overlap=False,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=50,
        # 'overlap_param_gather_with_optimizer_step' is set automatically. Added here for user's knowledge
        overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
    )
    recipe.trainer.callbacks.extend(
        [
            garbage_collection_callback,
            mcomm_overlap_callback,
        ]
    )

    recipe.trainer.plugins.grad_reduce_in_fp32 = False
    recipe.optim.config.use_precision_aware_optimizer = False

    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 4) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def run_pretraining():
    tp, pp, cp = (1, 1, 1)
    dp = int(8 / tp / pp / cp)
    micro_batch_size = 2
    global_batch_size = dp * micro_batch_size
    recipe = pretrain_recipe(
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        tensor_parallelism=tp,
        pipeline_parallelism=pp,
        context_parallelism=cp,
        sequence_parallelism=True,
        num_nodes=1,
        num_gpus_per_node=8,
        max_steps=20, # Setting a small value for the quickstart
    )

    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    run.run(recipe, executor=executor, name="mistral_7b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()
