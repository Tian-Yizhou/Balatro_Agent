"""Action-masking RLModule for RLlib's new API stack (PPO + Torch).

Implements the same pattern as Ray's ``ActionMaskingTorchRLModule`` example,
but packaged for direct use with :class:`BalatroRLlibEnv`.  The module expects
observations shaped as::

    {"action_mask": Tensor, "observations": Tensor}

Invalid actions are masked by setting their logits to ``-inf`` before the
categorical distribution is constructed.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import gymnasium as gym

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class _ActionMaskingBase(RLModule):
    """Base mixin that strips the action mask out of the Dict observation.

    Subclass alongside a framework-specific PPO module (e.g.
    :class:`PPOTorchRLModule`) using MRO.
    """

    @override(RLModule)
    def __init__(
        self,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[Union[dict, DefaultModelConfig]] = None,
        catalog_class=None,
        **kwargs,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                "ActionMaskingRLModule requires a gym.spaces.Dict observation "
                "space with keys 'action_mask' and 'observations'."
            )

        # Store the full space (with mask) for later reference, but pass
        # only the inner observations to the parent so it builds the correct
        # network topology.
        self.observation_space_with_mask = observation_space
        self.observation_space = observation_space["observations"]
        self._checked_observations = False

        super().__init__(
            observation_space=self.observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )


class ActionMaskingTorchRLModule(_ActionMaskingBase, PPOTorchRLModule):
    """PPO Torch RL module with action masking support.

    Use via ``RLModuleSpec(module_class=ActionMaskingTorchRLModule, ...)``.
    """

    @override(PPOTorchRLModule)
    def setup(self):
        super().setup()
        # Restore the full Dict observation space after the parent has used
        # the inner space to build its networks.
        self.observation_space = self.observation_space_with_mask

    # ---- forward passes ------------------------------------------------

    @override(PPOTorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_inference(batch, **kwargs)
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_exploration(batch, **kwargs)
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        outs = super()._forward_train(batch, **kwargs)
        return self._mask_action_logits(outs, batch["action_mask"])

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, TensorType], embeddings=None
    ):
        if isinstance(batch[Columns.OBS], dict):
            action_mask, batch = self._preprocess_batch(batch)
            batch["action_mask"] = action_mask
        return super().compute_values(batch, embeddings)

    # ---- helpers -------------------------------------------------------

    def _preprocess_batch(
        self, batch: Dict[str, TensorType]
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Extract the action mask and unwrap observations."""
        self._check_batch(batch)
        action_mask = batch[Columns.OBS].pop("action_mask")
        batch[Columns.OBS] = batch[Columns.OBS].pop("observations")
        return action_mask, batch

    @staticmethod
    def _mask_action_logits(
        batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        """Set logits of invalid actions to -inf."""
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        batch[Columns.ACTION_DIST_INPUTS] += inf_mask
        return batch

    def _check_batch(self, batch: Dict[str, TensorType]) -> None:
        if not self._checked_observations:
            obs = batch[Columns.OBS]
            if "action_mask" not in obs:
                raise ValueError(
                    "No 'action_mask' in observation dict. "
                    "Use BalatroRLlibEnv to wrap the environment."
                )
            if "observations" not in obs:
                raise ValueError(
                    "No 'observations' in observation dict. "
                    "Use BalatroRLlibEnv to wrap the environment."
                )
            self._checked_observations = True
