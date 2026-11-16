import logging
from abc import ABC, abstractmethod

from PIL import Image

log = logging.getLogger(__name__)


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_pil: Image.Image, text_query: str) -> list:
        """Detect objects matching text_query in image_pil.

        Returns list of (label: str, score: float, box: [x1,y1,x2,y2]) in pixel coords.
        """


class GroundingDINODetector(BaseDetector):
    """Zero-shot detector backed by GroundingDINO via HuggingFace Transformers."""

    def __init__(self, model_id: str, device: str, box_thr: float = 0.3, text_thr: float = 0.25):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        log.info(f"Loading GroundingDINO: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device
        self.box_thr = box_thr
        self.text_thr = text_thr
        self._torch = torch

    def detect(self, image_pil, text_query):
        text_query = text_query.strip().lower().rstrip('.') + '.'
        inputs = self.processor(images=image_pil, text=text_query, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        H, W = image_pil.size[1], image_pil.size[0]
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_thr,
            text_threshold=self.text_thr,
            target_sizes=[(H, W)],
        )[0]
        return [
            (label, float(score), box.cpu().tolist())
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"])
        ]


class SAM3Detector(BaseDetector):
    """Text-prompted detector backed by SAM 3 via HuggingFace Transformers."""

    def __init__(self, device: str, box_thr: float = 0.0, text_thr: float = 0.25, model_id: str = "facebook/sam3"):
        from transformers import Sam3Processor, Sam3Model
        import torch
        log.info(f"Loading SAM 3: {model_id}")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device
        self.box_thr = box_thr
        self._torch = torch

    def detect(self, image_pil, text_query):
        """Returns only the highest-scoring instance, as (label, score, box, mask), so its SAM 3 mask can be used directly."""
        inputs = self.processor(images=image_pil, text=text_query.strip(), return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.box_thr,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        if len(results["scores"]) == 0:
            return []
        best = int(results["scores"].argmax())
        return [(text_query, float(results["scores"][best]), results["boxes"][best].cpu().tolist(),
                 results["masks"][best].cpu().numpy().astype(bool))]


# ── Registry ──────────────────────────────────────────────────────────────────

_DETECTOR_REGISTRY = {
    "gdino": GroundingDINODetector,
    "sam3": SAM3Detector,
}


def build_detector(name: str, device: str, box_thr: float = 0.3, text_thr: float = 0.25,
                   model_id: str = "IDEA-Research/grounding-dino-base") -> BaseDetector:
    """Instantiate a detector backend by name.

    Supported names: gdino, sam3
    """
    if name not in _DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector backend '{name}'. Available: {list(_DETECTOR_REGISTRY)}")
    kwargs = {}
    if name == "gdino":
        kwargs["model_id"] = model_id
    return _DETECTOR_REGISTRY[name](device=device, box_thr=box_thr, text_thr=text_thr, **kwargs)
