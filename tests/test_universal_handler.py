import pytest
import torch
import torch.nn as nn

from src.handlers.universal_handler import UniversalHandler


class ModuleListModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )


class CacheAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)])

    def forward(self, hidden, *, use_cache):
        for layer in self.layers:
            hidden = layer(hidden)
        cache = tuple(hidden.detach() for _ in self.layers) if use_cache else None
        return hidden, cache


@pytest.mark.parametrize("model_cls", [ModuleListModel, SequentialModel])
def test_remove_layer_inplace_updates_metadata_and_preserves_container_type(model_cls):
    model = model_cls()
    original_type = type(model.layers)
    handler = UniversalHandler(model, verbose=False)

    pruned = handler.remove_layer("main", 1, inplace=True)

    assert pruned is model
    assert isinstance(model.layers, original_type)
    assert handler.component_info("main").layers is model.layers
    assert handler.component_info("main").count == 2
    assert len(handler.get_layers("main")) == 2


@pytest.mark.parametrize("model_cls", [ModuleListModel, SequentialModel])
def test_remove_layers_copy_preserves_original_metadata_and_container_type(model_cls):
    model = model_cls()
    original_layers = model.layers
    original_type = type(original_layers)
    handler = UniversalHandler(model, verbose=False)

    pruned = handler.remove_layers("main", [1], inplace=False)

    assert pruned is not model
    assert isinstance(pruned.layers, original_type)
    assert len(pruned.layers) == 2
    assert model.layers is original_layers
    assert len(model.layers) == 3
    assert handler.component_info("main").layers is original_layers
    assert handler.component_info("main").count == 3


@pytest.mark.parametrize(
    "indices",
    [
        [True],
        [1.0],
        ["1"],
        [-1],
        [1, 1],
        [3],
    ],
)
def test_remove_layers_rejects_invalid_indices_before_mutation(indices):
    model = ModuleListModel()
    original_layers = model.layers
    handler = UniversalHandler(model, verbose=False)

    with pytest.raises(ValueError):
        handler.remove_layers("main", indices, inplace=True)

    assert model.layers is original_layers
    assert len(model.layers) == 3
    assert handler.component_info("main").layers is original_layers
    assert handler.component_info("main").count == 3


@pytest.mark.parametrize("index", [True, 1.0, "1", -1, 3])
def test_remove_layer_rejects_invalid_index_before_mutation(index):
    model = SequentialModel()
    original_layers = model.layers
    handler = UniversalHandler(model, verbose=False)

    with pytest.raises(ValueError):
        handler.remove_layer("main", index, inplace=True)

    assert model.layers is original_layers
    assert len(model.layers) == 3
    assert handler.component_info("main").layers is original_layers
    assert handler.component_info("main").count == 3


def test_pruned_output_is_identical_with_cache_enabled_or_disabled():
    torch.manual_seed(1234)
    model = CacheAwareModel().eval()
    handler = UniversalHandler(model, verbose=False)
    handler.remove_layer("main", 1, inplace=True)
    hidden = torch.randn(2, 3, 4)

    without_cache, _ = model(hidden, use_cache=False)
    with_cache, cache = model(hidden, use_cache=True)

    torch.testing.assert_close(with_cache, without_cache, rtol=0.0, atol=0.0)
    assert len(cache) == 2
