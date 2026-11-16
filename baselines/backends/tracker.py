import logging

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


class SAM3VideoPredictor:
    """SAM 3 video tracker (HuggingFace Transformers) with the SAM2 video predictor calls used by complete_obj_tracks.py."""

    def __init__(self, device: str, model_id: str = "facebook/sam3"):
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        log.info(f"Loading SAM 3 tracker: {model_id}")
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
        self.model = Sam3TrackerVideoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device

    def init_state(self, frame_paths, **kwargs):
        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        return self.processor.init_video_session(
            video=frames, inference_device=self.device,
            video_storage_device="cpu", inference_state_device="cpu",
        )

    def reset_state(self, session):
        session.reset_tracking_data()

    def add_new_mask(self, session, frame_idx, obj_id, mask):
        self.processor.add_inputs_to_inference_session(
            session, frame_idx=frame_idx, obj_ids=obj_id, input_masks=np.asarray(mask, dtype=np.float32),
        )
        self.start_frame = frame_idx  # SAM2 starts from the prompted frame; SAM 3 needs it passed explicitly

    @torch.no_grad()
    def propagate_in_video(self, session, reverse=False):
        """Yield (frame_idx, obj_ids, logits) like SAM2, with logits (num_objs, 1, H, W) at video resolution."""
        size = [[session.video_height, session.video_width]]
        if reverse and self.start_frame == 0:
            # SAM2 still yields frame 0 when propagating backward from it; SAM 3 yields nothing
            outputs = [self.model(session, frame_idx=0, reverse=True)]
        else:
            outputs = self.model.propagate_in_video_iterator(session, start_frame_idx=self.start_frame, reverse=reverse)
        for out in outputs:
            logits = self.processor.post_process_masks([out.pred_masks], original_sizes=size, binarize=False)[0]
            yield out.frame_idx, session.obj_ids, logits
