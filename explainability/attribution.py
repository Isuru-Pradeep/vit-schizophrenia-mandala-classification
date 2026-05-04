from typing import Optional

import torch
import torch.nn.functional as F


def _raise_if_cancelled(cancel_callback=None):
    if cancel_callback is not None and cancel_callback():
        raise InterruptedError("Task stopped by user.")


def _prepare_target(logits: torch.Tensor, target_class: Optional[int]) -> torch.Tensor:
    if target_class is None:
        target_index = int(torch.argmax(logits, dim=1).item())
    else:
        target_index = int(target_class)
    return logits[:, target_index], target_index


def _normalize_map(score_map: torch.Tensor) -> torch.Tensor:
    score_map = score_map.detach()
    score_map = score_map - score_map.min()
    denom = score_map.max()
    if float(denom) > 0:
        score_map = score_map / denom
    return score_map.clamp(0.0, 1.0)


def _log_map_stats(method_name: str, score_map: torch.Tensor):
    print(
        f"[{method_name}] "
        f"min={float(score_map.min()):.6f}, "
        f"max={float(score_map.max()):.6f}, "
        f"mean={float(score_map.mean()):.6f}, "
        f"std={float(score_map.std()):.6f}"
    )


def _get_patch_grid(model, image_tensor: torch.Tensor):
    patch_embed = getattr(model.backbone, "patch_embed", None)
    if patch_embed is not None:
        embedded = patch_embed(image_tensor)

        if embedded.dim() == 4:
            patch_h, patch_w = embedded.shape[-2:]
            return int(patch_h), int(patch_w)

        if embedded.dim() == 3:
            num_patches = embedded.shape[1]
            grid_size = int(num_patches ** 0.5)
            if grid_size * grid_size == num_patches:
                return int(grid_size), int(grid_size)

    image_size = image_tensor.shape[-1]
    patch_size = 16
    if hasattr(model.backbone, "patch_embed") and hasattr(model.backbone.patch_embed, "patch_size"):
        patch_shape = model.backbone.patch_embed.patch_size
        patch_size = patch_shape[0] if isinstance(patch_shape, tuple) else int(patch_shape)
    grid_size = image_size // patch_size
    return int(grid_size), int(grid_size)


def _get_vit_gradcam_hook_module(model):
    blocks = getattr(model.backbone, "blocks", None)
    if blocks:
        last_block = blocks[-1]
        for attribute_name in ["norm1", "norm2"]:
            hook_module = getattr(last_block, attribute_name, None)
            if hook_module is not None:
                return hook_module

    hook_module = getattr(model.backbone, "norm", None)
    if hook_module is not None:
        return hook_module

    raise ValueError("ViT Grad-CAM could not find a suitable transformer normalization layer.")


def compute_vit_gradcam(
    model,
    image_tensor: torch.Tensor,
    completion_time_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    cancel_callback=None,
):
    _raise_if_cancelled(cancel_callback)
    model.eval()
    image = image_tensor.clone().detach()
    time_value = completion_time_tensor.clone().detach()

    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    hook_module = _get_vit_gradcam_hook_module(model)

    handle_fwd = hook_module.register_forward_hook(forward_hook)
    handle_bwd = hook_module.register_full_backward_hook(backward_hook)

    logits = model(image, time_value)
    _raise_if_cancelled(cancel_callback)
    target_logit, target_index = _prepare_target(logits, target_class)

    model.zero_grad(set_to_none=True)
    target_logit.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    token_activations = activations["value"]
    token_gradients = gradients["value"]

    if token_activations.dim() != 3 or token_gradients.dim() != 3:
        raise ValueError("ViT Grad-CAM expects token activations from the transformer norm layer.")

    patch_activations = token_activations[:, 1:, :]
    patch_gradients = token_gradients[:, 1:, :]
    weights = patch_gradients.mean(dim=1)
    cam = torch.einsum("bpc,bc->bp", patch_activations, weights).squeeze(0)

    grid_h, grid_w = _get_patch_grid(model, image_tensor)
    cam = cam.reshape(grid_h, grid_w)
    relu_cam = F.relu(cam)

    # Some ViT backbones produce mostly negative CAM scores at this layer.
    # If ReLU wipes out almost the whole map, fall back to the absolute
    # activation-gradient interaction to preserve a visible attribution signal.
    if float(relu_cam.max()) <= 1e-8 or float((relu_cam > 0).sum()) <= 1:
        cam = (patch_activations * patch_gradients).sum(dim=-1).abs().squeeze(0).reshape(grid_h, grid_w)
    else:
        cam = relu_cam

    print(f"[gradcam] hook={hook_module.__class__.__name__}")
    _log_map_stats("gradcam_raw", cam)

    cam = _normalize_map(cam)
    _log_map_stats("gradcam_norm", cam)
    probabilities = torch.softmax(logits.detach(), dim=1).squeeze(0)

    return {
        "method": "gradcam",
        "target_class": target_index,
        "probabilities": probabilities.cpu(),
        "score_map": cam.cpu(),
    }


def compute_attention_visualization(
    model,
    image_tensor: torch.Tensor,
    completion_time_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    cancel_callback=None,
):
    _raise_if_cancelled(cancel_callback)
    model.eval()
    image = image_tensor.clone().detach()
    time_value = completion_time_tensor.clone().detach()

    attention_maps = []
    handles = []

    for block in getattr(model.backbone, "blocks", []):
        if hasattr(block, "attn") and hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False
        if hasattr(block, "attn") and hasattr(block.attn, "attn_drop"):
            handles.append(
                block.attn.attn_drop.register_forward_hook(
                    lambda _, __, output: attention_maps.append(output.detach())
                )
            )

    logits = model(image, time_value)
    _raise_if_cancelled(cancel_callback)
    _, target_index = _prepare_target(logits, target_class)

    for handle in handles:
        handle.remove()

    if not attention_maps:
        raise ValueError("Attention visualization could not capture ViT attention maps.")

    num_tokens = attention_maps[0].shape[-1]
    rollout = torch.eye(num_tokens, device=image.device).unsqueeze(0)

    for attention in attention_maps:
        _raise_if_cancelled(cancel_callback)
        attn_mean = attention.mean(dim=1)
        attn_mean = attn_mean + torch.eye(num_tokens, device=attn_mean.device).unsqueeze(0)
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(attn_mean, rollout)

    cls_attention = rollout[:, 0, 1:].squeeze(0)
    grid_h, grid_w = _get_patch_grid(model, image_tensor)
    cls_attention = cls_attention.reshape(grid_h, grid_w)
    _log_map_stats("attention_raw", cls_attention)
    cls_attention = _normalize_map(cls_attention)
    _log_map_stats("attention_norm", cls_attention)
    probabilities = torch.softmax(logits.detach(), dim=1).squeeze(0)

    return {
        "method": "attention_visualization",
        "target_class": target_index,
        "probabilities": probabilities.cpu(),
        "score_map": cls_attention.cpu(),
    }


def compute_saliency_map(
    model,
    image_tensor: torch.Tensor,
    completion_time_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    cancel_callback=None,
):
    model.eval()
    time_value = completion_time_tensor.clone().detach()
    num_samples = 8
    noise_sigma = 0.08
    accumulated = torch.zeros_like(image_tensor.squeeze(0)[0], device=image_tensor.device)
    target_index = None
    probabilities = None

    for _ in range(num_samples):
        _raise_if_cancelled(cancel_callback)
        noisy_image = image_tensor.clone().detach()
        noise = torch.randn_like(noisy_image) * noise_sigma
        noisy_image = (noisy_image + noise).requires_grad_(True)

        logits = model(noisy_image, time_value)
        target_logit, current_target_index = _prepare_target(logits, target_class)
        if target_index is None:
            target_index = current_target_index
            probabilities = torch.softmax(logits.detach(), dim=1).squeeze(0)

        model.zero_grad(set_to_none=True)
        target_logit.backward()
        gradients = noisy_image.grad.detach().abs().max(dim=1)[0].squeeze(0)
        accumulated += gradients

    saliency = accumulated / float(num_samples)
    _log_map_stats("saliency_raw", saliency)
    saliency = _normalize_map(saliency)
    _log_map_stats("saliency_norm", saliency)

    return {
        "method": "saliency",
        "target_class": target_index,
        "probabilities": probabilities.cpu(),
        "score_map": saliency.cpu(),
    }


def compute_integrated_gradients(
    model,
    image_tensor: torch.Tensor,
    completion_time_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    steps: int = 32,
    cancel_callback=None,
):
    model.eval()
    image = image_tensor.clone().detach()
    time_value = completion_time_tensor.clone().detach()
    baseline = image.mean(dim=(-1, -2), keepdim=True).expand_as(image).detach()

    with torch.no_grad():
        initial_logits = model(image, time_value)
    _, target_index = _prepare_target(initial_logits, target_class)

    total_gradients = torch.zeros_like(image)

    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=image.device)[1:]:
        _raise_if_cancelled(cancel_callback)
        scaled = baseline + alpha * (image - baseline)
        scaled.requires_grad_(True)

        logits = model(scaled, time_value)
        target_logit = logits[:, target_index]

        model.zero_grad(set_to_none=True)
        target_logit.backward()
        total_gradients += scaled.grad.detach()

    avg_gradients = total_gradients / max(steps, 1)
    integrated = (image - baseline) * avg_gradients
    score_map = integrated.abs().max(dim=1)[0].squeeze(0)
    _log_map_stats("integrated_gradients_raw", score_map)
    score_map = _normalize_map(score_map)
    _log_map_stats("integrated_gradients_norm", score_map)
    probabilities = torch.softmax(initial_logits.detach(), dim=1).squeeze(0)

    return {
        "method": "integrated_gradients",
        "target_class": target_index,
        "probabilities": probabilities.cpu(),
        "score_map": score_map.cpu(),
    }


def compute_occlusion_sensitivity(
    model,
    image_tensor: torch.Tensor,
    completion_time_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    patch_size: int = 32,
    stride: int = 16,
    cancel_callback=None,
):
    _raise_if_cancelled(cancel_callback)
    model.eval()
    image = image_tensor.clone().detach()
    time_value = completion_time_tensor.clone().detach()

    with torch.no_grad():
        base_logits = model(image, time_value)
    _, target_index = _prepare_target(base_logits, target_class)
    base_probs = torch.softmax(base_logits.detach(), dim=1)
    base_score = float(base_probs[0, target_index].item())

    _, _, height, width = image.shape
    score_map = torch.zeros((height, width), device=image.device)
    counts = torch.zeros((height, width), device=image.device)

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            _raise_if_cancelled(cancel_callback)
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)

            occluded = image.clone()
            occluded[:, :, top:bottom, left:right] = 0.0

            with torch.no_grad():
                logits = model(occluded, time_value)
                probs = torch.softmax(logits, dim=1)
                drop = base_score - float(probs[0, target_index].item())

            score_map[top:bottom, left:right] += drop
            counts[top:bottom, left:right] += 1.0

    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    score_map = score_map / counts
    _log_map_stats("occlusion_raw", score_map)
    score_map = _normalize_map(score_map)
    _log_map_stats("occlusion_norm", score_map)

    return {
        "method": "occlusion",
        "target_class": target_index,
        "probabilities": base_probs.squeeze(0).cpu(),
        "score_map": score_map.cpu(),
    }
