"""Batched, differentiable tensor execution of the complete creative graph."""

import copy

import torch

from .graph import validate_graph
from .providers import PipelineRegistry, node_seed, validate_seed


GLYPH_ROWS = (14, 17, 17, 31, 17, 17, 17)


def validate_inputs(upl, coordinates):
    if upl.ndim != 2 or upl.shape[1] != 4:
        raise ValueError("UPL must have shape [batch,4]")
    if coordinates.ndim != 2 or coordinates.shape != (upl.shape[0], 2):
        raise ValueError("Coordinates must have shape [batch,2]")
    if upl.shape[0] == 0 or not upl.is_floating_point() or not coordinates.is_floating_point():
        raise ValueError("Inputs must be nonempty floating-point tensors")
    if upl.dtype != coordinates.dtype or upl.device != coordinates.device:
        raise ValueError("UPL and coordinates must share dtype and device")
    for values in (upl, coordinates):
        if not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Inputs must be finite and normalized to [0,1]")


def brand_masks(coordinates, parameters):
    horizontal = coordinates[:, 0]
    vertical = coordinates[:, 1]
    origin_horizontal = parameters["x"]
    origin_vertical = parameters["y"]
    cell = parameters["cell"]
    inside = (
        (horizontal >= origin_horizontal)
        & (horizontal < origin_horizontal + 7 * cell)
        & (vertical >= origin_vertical)
        & (vertical < origin_vertical + 9 * cell)
    )
    columns = torch.floor((horizontal - origin_horizontal) / cell).to(torch.int64)
    rows = torch.floor((vertical - origin_vertical) / cell).to(torch.int64)
    row_bits = torch.tensor(GLYPH_ROWS, device=coordinates.device, dtype=torch.int64)
    bits = row_bits[(rows - 1).clamp(0, 6)]
    shifts = (5 - columns).clamp(0, 4)
    foreground = (
        inside
        & (columns >= 1)
        & (columns <= 5)
        & (rows >= 1)
        & (rows <= 7)
        & (((bits >> shifts) & 1) != 0)
    )
    return inside, foreground


def apply_brand(image, coordinates, parameters):
    inside, foreground = brand_masks(coordinates, parameters)
    color = coordinates.new_tensor(parameters["color"])
    background = coordinates.new_tensor(parameters["background"])
    plate = torch.where(foreground[:, None], color, background)
    return torch.where(inside[:, None], plate, image)


def pixel_coordinates(width, height, device="cpu"):
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("Raster dimensions must be integers")
    if not isinstance(width, int) or not isinstance(height, int) or not 1 <= width <= 512 or not 1 <= height <= 512:
        raise ValueError("Raster dimensions must be integers between one and 512")
    horizontal = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    vertical = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    vertical_grid, horizontal_grid = torch.meshgrid(vertical, horizontal, indexing="ij")
    return torch.stack((horizontal_grid.reshape(-1), vertical_grid.reshape(-1)), dim=1)


def brand_probe_coordinates(parameters, device="cpu"):
    horizontal = (torch.arange(7, device=device, dtype=torch.float32) + 0.5) * parameters["cell"] + parameters["x"]
    vertical = (torch.arange(9, device=device, dtype=torch.float32) + 0.5) * parameters["cell"] + parameters["y"]
    vertical_grid, horizontal_grid = torch.meshgrid(vertical, horizontal, indexing="ij")
    return torch.stack((horizontal_grid.reshape(-1), vertical_grid.reshape(-1)), dim=1)


class TensorTeacher:
    def __init__(self, graph):
        self.nodes = validate_graph(graph)
        if graph["version"] != "angelos.graph.v1":
            raise ValueError("TensorTeacher/compile_teacher cannot execute graph.v2 semantic UPLs; legacy operators are not migrated")
        if any(node["op"] in {"prompt", "generate_text", "generate_image", "generate_video", "frame"} for node in self.nodes):
            raise ValueError("Coordinate teacher / strict ATen compiler cannot execute generative nodes. Use RasterTeacher or python -m angelos_ml.dataset with pinned local models; no procedural fallback or compiled-generator claim is supported.")
        self.graph = copy.deepcopy(graph)
        brands = [node["params"] for node in self.nodes if node["op"] == "brand"]
        self.brand = copy.deepcopy(brands[0]) if brands else None

    def color_at(self, upl, coordinates, validate=True):
        if validate:
            validate_inputs(upl, coordinates)
        horizontal = coordinates[:, 0:1]
        vertical = coordinates[:, 1:2]
        values = {}
        for node in self.nodes:
            operation = node["op"]
            parameters = node["params"]
            inputs = {port: values[source] for port, source in node["inputs"].items()}
            if operation == "upl":
                result = upl
            elif operation == "feature":
                index = int(parameters["index"])
                result = inputs["input"][:, index:index + 1]
            elif operation == "constant":
                result = torch.full_like(horizontal, parameters["value"])
            elif operation == "gradient":
                amount = (0.65 * horizontal + parameters["angle"] * (vertical - 0.5) + 0.35 * inputs["condition"]).clamp(0, 1)
                low = coordinates.new_tensor(parameters["low"])
                high = coordinates.new_tensor(parameters["high"])
                result = low + amount * (high - low)
            elif operation == "disk":
                center_horizontal = parameters["center"][0] + 0.12 * (inputs["condition"] - 0.5)
                displacement = torch.cat((horizontal - center_horizontal, vertical - parameters["center"][1]), dim=1)
                distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
                alpha = ((parameters["radius"] - distance) / parameters["softness"] + 0.5).clamp(0, 1)
                color = coordinates.new_tensor(parameters["color"])
                result = inputs["image"] + alpha * (color - inputs["image"])
            elif operation == "exposure":
                multiplier = 1 + parameters["amount"] * (2 * inputs["condition"] - 1)
                result = (inputs["image"] * multiplier).clamp(0, 1)
            elif operation == "vignette":
                distance = (2 * ((horizontal - 0.5).square() + (vertical - 0.5).square())).clamp(0, 1)
                result = inputs["image"] * (1 - parameters["strength"] * distance)
            elif operation == "brand":
                result = apply_brand(inputs["image"], coordinates, parameters)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            values[node["id"]] = result
        return values[self.graph["output"]]

    __call__ = color_at


class RasterTeacher:
    """Execute a complete DAG at native raster resolution before any resizing."""

    def __init__(self, graph, registry=None):
        self.nodes = validate_graph(graph)
        if graph["version"] != "angelos.graph.v1":
            raise ValueError("RasterTeacher cannot execute graph.v2 semantic UPLs; supply a frozen dataset, not fabricated teacher execution")
        self.graph = copy.deepcopy(graph)
        self.registry = registry if registry is not None else PipelineRegistry()
        self.last_provenance = []

    @torch.inference_mode()
    def render(self, upl, *, seed, width=256, height=256):
        profiles = torch.as_tensor(upl, dtype=torch.float32, device=self.registry.device)
        if profiles.ndim == 1:
            profiles = profiles.unsqueeze(0)
        validate_inputs(profiles, profiles.new_zeros((profiles.shape[0], 2)))
        pixel_coordinates(width, height, device=profiles.device)
        if isinstance(seed, (list, tuple)):
            if len(seed) != len(profiles):
                raise ValueError("Supply one teacher seed per UPL profile")
            seeds = [validate_seed(value) for value in seed]
        else:
            validate_seed(seed)
            seeds = [validate_seed(seed + index) for index in range(len(profiles))]
        self.last_provenance = []
        images = []
        for profile, sample_seed in zip(profiles, seeds):
            image, provenance = self._render_profile(profile, sample_seed, width, height)
            images.append(image)
            self.last_provenance.append(provenance)
        return torch.stack(images)

    def _render_profile(self, profile, seed, width, height):
        values = {}
        provenance = {"teacherSeed": seed, "deterministicTensorDevice": str(profile.device), "generations": []}
        for node in self.nodes:
            operation = node["op"]
            parameters = node["params"]
            inputs = {port: values[source] for port, source in node["inputs"].items()}
            if operation == "upl":
                result = profile
            elif operation == "feature":
                result = inputs["input"][int(parameters["index"])].item()
            elif operation == "constant":
                result = parameters["value"]
            elif operation == "prompt":
                condition = inputs["condition"]
                text = parameters["low"] if condition < parameters["threshold"] else parameters["high"]
                result = text.replace("{condition}", f"{condition:.6f}")
            elif operation == "generate_text":
                derived_seed = node_seed(seed, node["id"])
                generated = self.registry.generate_text(parameters["model"], inputs["prompt"], parameters, derived_seed)
                result = generated["text"]
                provenance["generations"].append({
                    "node": node["id"], "seed": derived_seed, "prompt": inputs["prompt"],
                    "model": self.registry.metadata(parameters["model"]), "params": copy.deepcopy(parameters), **generated,
                })
            elif operation in {"generate_image", "generate_video"}:
                derived_seed = node_seed(seed, node["id"])
                text_source = any(source["id"] == node["inputs"]["prompt"] and source["op"] == "generate_text" for source in self.nodes)
                result = self.registry.generate(parameters["model"], inputs["prompt"], parameters, derived_seed, strict_prompt=text_source)
                provenance["generations"].append({
                    "node": node["id"], "seed": derived_seed, "prompt": inputs["prompt"],
                    "model": self.registry.metadata(parameters["model"]), "params": copy.deepcopy(parameters),
                })
            elif operation == "frame":
                result = inputs["video"][int(parameters["index"])]
            else:
                image = inputs.get("image")
                raster_height, raster_width = image.shape[1:] if image is not None else (height, width)
                coordinates = pixel_coordinates(raster_width, raster_height, device=profile.device)
                horizontal = coordinates[:, 0:1]
                vertical = coordinates[:, 1:2]
                pixels = image.permute(1, 2, 0).reshape(-1, 3) if image is not None else None
                if operation == "gradient":
                    amount = (0.65 * horizontal + parameters["angle"] * (vertical - 0.5) + 0.35 * inputs["condition"]).clamp(0, 1)
                    low = coordinates.new_tensor(parameters["low"])
                    high = coordinates.new_tensor(parameters["high"])
                    result = low + amount * (high - low)
                elif operation == "disk":
                    center = parameters["center"][0] + 0.12 * (inputs["condition"] - 0.5)
                    displacement = torch.cat((horizontal - center, vertical - parameters["center"][1]), dim=1)
                    distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
                    alpha = ((parameters["radius"] - distance) / parameters["softness"] + 0.5).clamp(0, 1)
                    result = pixels + alpha * (coordinates.new_tensor(parameters["color"]) - pixels)
                elif operation == "exposure":
                    result = (pixels * (1 + parameters["amount"] * (2 * inputs["condition"] - 1))).clamp(0, 1)
                elif operation == "vignette":
                    distance = (2 * ((horizontal - 0.5).square() + (vertical - 0.5).square())).clamp(0, 1)
                    result = pixels * (1 - parameters["strength"] * distance)
                elif operation == "brand":
                    result = apply_brand(pixels, coordinates, parameters)
                else:
                    raise ValueError(f"Unsupported raster operation: {operation}")
                result = result.reshape(raster_height, raster_width, 3).permute(2, 0, 1).contiguous()
            values[node["id"]] = result
        return values[self.graph["output"]], provenance
