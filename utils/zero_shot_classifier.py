from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch



def build_zero_shot_classifier(
        text_model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names in batches
    Args:
        text_model: CLIP text encoder
        tokenizer: CLIP tokenizer
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [template.format(c.lower()) for c in batch_classnames for template in templates]
        texts = tokenizer(texts, padding="max_length", max_length=64, return_tensors="pt").input_ids.to(device)
        class_embeddings = text_model(texts).pooler_output
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        return class_embeddings

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=0)
        else:
            zeroshot_weights = _process_batch(classnames)
        zeroshot_weights = F.normalize(zeroshot_weights, dim=-1)
    return zeroshot_weights
