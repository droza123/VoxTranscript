"""
torch >= 2.6 defaults torch.load() to weights_only=True, which refuses any class
that isn't explicitly allowlisted. The pyannote checkpoints we load (whisperx's
VAD model and the diarization segmentation/embedding models) pickle a handful of
omegaconf / pyannote / torch classes in their hyperparameters, so they fail with
"Weights only load failed".

Rather than turning the check off, allowlist exactly the classes those
checkpoints contain. Import this module before any pyannote model is loaded.
"""
import collections
import logging
import typing

import torch


def _allow_pyannote_checkpoint_globals():
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return  # torch < 2.4: weights_only defaults to False, nothing to do

    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode
    from pyannote.audio.core.model import Introspection
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from torch.torch_version import TorchVersion

    add_safe_globals([
        ListConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        TorchVersion,
        Introspection,
        Specifications,
        Problem,
        Resolution,
        typing.Any,
        collections.defaultdict,
        list,
        dict,
        int,
    ])
    # Module logger, not logging.info(): a root-level logging call with no handlers
    # yet would run basicConfig() and turn transcriber.setup_logging() into a no-op.
    logging.getLogger(__name__).info("Allowlisted pyannote checkpoint classes for torch.load(weights_only=True)")


_allow_pyannote_checkpoint_globals()
