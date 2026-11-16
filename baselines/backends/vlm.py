import os, re, json, base64, io, logging

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are analyzing a video to detect physical transformation events.
A transformation event is one of:
- decomposition: one object splits, opens, or separates into two distinct objects
- composition: two objects combine, merge, or are inserted into one another

Output ONLY a valid JSON array (no markdown, no explanation). Each element must have:
  "frame_index": int (index into the frames you were shown, 0-based),
  "type": "decomposition" or "composition",
  "parent_object": str (the object that was whole before, or the container that receives),
  "parent_object_loc": [x1,y1,x2,y2] normalized to [0,1] relative to frame width/height, (bounding box of the parent object in the frame where the event occurs),
  "child_object": str (the object that emerged / was extracted / was inserted),
  "child_object_loc":  [x1,y1,x2,y2] normalized to [0,1] relative to frame width/height (bounding box of the child object in the frame where the event occurs),
  "action": str (one verb describing the event, e.g. "separate", "pour", "insert")

The object descriptions should be optimized for down-stream open-vocabulary detection, so be specific and descriptive, but as concise as possible (e.g. "red liquid in glass" rather than "the thing on the left"). 
The action should be a single verb that best describes the transformation (e.g. "separate", "pour", "insert", "open", "close", "break", etc.).
If no transformation events occur, output []. 

IMPORTANT: parent_object and child_object must be spatially distinct objects in the frame.
Do not report state changes of a single object (e.g. sealed→opened bag).
A valid event requires two different bounding boxes that do not heavily overlap.
"""


# Same schema as SYSTEM_PROMPT, but events are localized by video timestamp instead of
# an index into sampled frames (for models with native video input, i.e. Gemini).
VIDEO_SYSTEM_PROMPT = """You are analyzing a video to detect physical transformation events.
A transformation event is one of:
- decomposition: one object splits, opens, or separates into two distinct objects
- composition: two objects combine, merge, or are inserted into one another

Output ONLY a valid JSON array (no markdown, no explanation). Each element must have:
  "timestamp": str ("MM:SS" time in the video at which the transformation first becomes clearly visible),
  "type": "decomposition" or "composition",
  "parent_object": str (the object that was whole before, or the container that receives),
  "parent_object_loc": [x1,y1,x2,y2] normalized to [0,1] relative to frame width/height, (bounding box of the parent object at the event timestamp),
  "child_object": str (the object that emerged / was extracted / was inserted),
  "child_object_loc":  [x1,y1,x2,y2] normalized to [0,1] relative to frame width/height (bounding box of the child object at the event timestamp),
  "action": str (one verb describing the event, e.g. "separate", "pour", "insert")

The object descriptions should be optimized for down-stream open-vocabulary detection, so be specific and descriptive, but as concise as possible (e.g. "red liquid in glass" rather than "the thing on the left").
The action should be a single verb that best describes the transformation (e.g. "separate", "pour", "insert", "open", "close", "break", etc.).
If no transformation events occur, output [].

IMPORTANT: parent_object and child_object must be spatially distinct objects in the frame.
Do not report state changes of a single object (e.g. sealed→opened bag).
A valid event requires two different bounding boxes that do not heavily overlap.
"""


def _encode_image(image_np):
    if image_np.max() <= 1.0:
        image_np = (image_np * 255)
    img = Image.fromarray(image_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8'), 'image/png'


def _img_payload(image_np, detail='low'):
    b64, mime = _encode_image(image_np)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}


def _txt_payload(text):
    return {"type": "text", "text": text}


def parse_timestamp(ts):
    """'MM:SS' / 'H:MM:SS' / numeric seconds -> float seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = [float(p) for p in str(ts).strip().split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def _parse_json_list(raw):
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        log.warning("Could not parse JSON array from VLM response; assuming no events.")
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e}; assuming no events.")
        return []


# ── Clients ───────────────────────────────────────────────────────────────────

# Both clients take content as OpenAI-style text / image_url payloads (built by _txt_payload / _img_payload here).

class OpenAIClient:
    """GPT via the OpenAI API (OPENAI_API_KEY), or Qwen via the HF router (HF_API_KEY), which uses the same API."""

    def __init__(self, model_name: str):
        import openai
        if model_name.startswith("Qwen"):
            self.client = openai.OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.environ.get("HF_API_KEY"),
            )
        else:
            self.client = openai.OpenAI()
        self.model_name = model_name

    def query(self, system_prompt: str, content: list, temp: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": content},
            ],
            temperature=temp,
        )
        return response.choices[0].message.content


class GeminiClient:
    """Gemini via the google-genai API (GEMINI_API_KEY)."""

    def __init__(self, model_name: str):
        from google import genai
        from google.genai import types
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            # retry 429s with backoff; Gemini rate limits are tight
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=8)),
        )
        self.model_name = model_name

    def query(self, system_prompt: str, content: list, temp: float = 0.0) -> str:
        from google.genai import types
        parts = []
        for item in content:
            if item["type"] == "text":
                parts.append(types.Part.from_text(text=item["text"]))
            elif item["type"] == "image_url":
                header, b64 = item["image_url"]["url"].split(",", 1)  # data:<mime>;base64,<data>
                mime = header[len("data:"):].split(";")[0]
                parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime))
            else:
                raise ValueError(f"Unsupported content type: {item['type']}")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=temp),
        )
        return response.text or ""

    def query_video(self, system_prompt: str, video_path: str, prompt_text: str, fps: float, temp: float = 0.0) -> str:
        """Query with native video input, sampled by the API at `fps`.

        Videos under the ~20MB inline request limit are sent as inline bytes; larger ones
        go through the Files API (uploaded, polled until ACTIVE, then referenced by URI).
        """
        import time
        from google.genai import types
        video_metadata = types.VideoMetadata(fps=fps)
        if os.path.getsize(video_path) < 19 * 1024 * 1024:
            video_part = types.Part(
                inline_data=types.Blob(data=open(video_path, "rb").read(), mime_type="video/mp4"),
                video_metadata=video_metadata,
            )
        else:
            upload = self.client.files.upload(file=video_path)
            while upload.state and upload.state.name == "PROCESSING":
                time.sleep(2)
                upload = self.client.files.get(name=upload.name)
            if upload.state and upload.state.name != "ACTIVE":
                raise RuntimeError(f"Video upload failed for {video_path}: {upload.state.name}")
            video_part = types.Part(
                file_data=types.FileData(file_uri=upload.uri, mime_type="video/mp4"),
                video_metadata=video_metadata,
            )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[video_part, types.Part.from_text(text=prompt_text)],
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=temp),
        )
        return response.text or ""


def build_vlm_client(model_name: str):
    """Return the API client for model_name; call client.query(system_prompt, content, temp)."""
    return GeminiClient(model_name) if model_name.startswith("gemini") else OpenAIClient(model_name)


class APIVLM:
    """Calls GPT (OpenAI API), Qwen (HF router) or Gemini (google-genai API)."""

    def __init__(self, model_name: str, temp: float = 0.0):
        self.client = build_vlm_client(model_name)
        self.model_name = model_name
        self.temp = temp

    def query_events(self, sampled_frames):
        """Query VLM with subsampled frames; return list of raw event dicts.

        Each dict must have: frame_index, type, parent_object, child_object, action.
        """
        content = []
        for i, frame_np in enumerate(sampled_frames):
            content.append(_txt_payload(f"Frame {i}:"))
            content.append(_img_payload(frame_np, detail='low'))
        content.append(_txt_payload(
            "Identify all physical transformation events (decomposition or composition) "
            "in the frames above. Output only a JSON array as specified."
        ))
        raw = self.client.query(SYSTEM_PROMPT, content, self.temp).strip()
        log.info(f"VLM response: {raw}")
        return _parse_json_list(raw)

    def query_events_video(self, video_path, fps=5):
        """Query with native video input; return list of raw event dicts with timestamps.

        Each dict must have: timestamp, type, parent_object, child_object, action.
        """
        if not isinstance(self.client, GeminiClient):
            raise ValueError(f"Native video input requires a Gemini model, got {self.model_name}")
        raw = self.client.query_video(
            VIDEO_SYSTEM_PROMPT, video_path,
            "Identify all physical transformation events (decomposition or composition) "
            "in this video. Output only a JSON array as specified.",
            fps=fps, temp=self.temp,
        ).strip()
        log.info(f"VLM response: {raw}")
        return _parse_json_list(raw)


def build_vlm(name: str, temp: float = 0.0) -> APIVLM:
    """Instantiate the VLM backend for model `name` (GPT, Qwen, or Gemini)."""
    return APIVLM(name, temp)
