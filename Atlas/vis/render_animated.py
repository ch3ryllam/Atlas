"""
Render an animated video that stacks the precomputed segmentation video on top of
the PDF timeline chart.  The timeline progressively reveals itself left-to-right
in sync with the video — content to the right of the current frame is hidden.

Usage:
python3 Atlas/vis/render_animated.py -c configs/default.yaml -d ours -i crackers
"""
import sys, argparse, math
import os.path as osp
import numpy as np
import cv2
import imageio.v3 as iio
from tqdm import tqdm

try:
    import fitz  # pymupdf
except ImportError:
    raise ImportError("pymupdf is required: pip install pymupdf")

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file, write_video

# Where the timeline sits on the page — must match render_pdf.py. The page width is read
# off the rendered image instead, and the time axis spans the whole clip in both files.
TL_MARGIN_LEFT = 950
TL_DRAW_W = 1350

def pdf_to_image(pdf_path):
    """Render first page of PDF to an RGB numpy array at 72 DPI (1 pt = 1 px)."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = img[..., :3]
    return img


def mask_future(pdf_base, frame_idx, fps, max_sec):
    """Return a copy of pdf_base with everything right of the current time hidden."""
    img = pdf_base.copy()
    x = int(TL_MARGIN_LEFT + ((frame_idx + 1) / fps / max_sec) * TL_DRAW_W)
    img[:, x:] = 255
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='configs/default.yaml')
    p.add_argument('-d', '--dataset', default='ours')
    p.add_argument('-i', '--instance', required=True)
    p.add_argument('--skip_fp', action='store_true')
    p.add_argument('--skip_temporal_alignment', action='store_true')
    args = p.parse_args()

    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    fps = data_cfg.fps
    instance = args.instance
    output_dataset = args.dataset
    if args.skip_fp: output_dataset += '_skip_fp'
    if args.skip_temporal_alignment: output_dataset += '_skip_temporal_alignment'
    vis_dir = osp.join(cfg.paths.visdir, output_dataset)

    video_path = osp.join(vis_dir, f'{instance}_final.mp4')
    pdf_path = osp.join(vis_dir, f'{instance}_state_graph.pdf')
    for path in (video_path, pdf_path):
        if not osp.exists(path):
            raise FileNotFoundError(f"Missing precomputed file: {path}\nRun Atlas/get_animate.py first.")

    print("Rendering PDF to image...")
    pdf_img = pdf_to_image(pdf_path)
    pdf_height = pdf_img.shape[0]

    print("Reading video frames...")
    video_frames = list(iio.imiter(video_path))
    if not video_frames:
        raise RuntimeError(f"Could not read frames from {video_path}")

    vid_w = video_frames[0].shape[1]
    pdf_scaled_h = int(pdf_height * vid_w / pdf_img.shape[1])
    max_sec = math.ceil((len(video_frames) - 1) / fps) or 1

    out_frames = []
    for frame_idx, vid_frame in enumerate(tqdm(video_frames, desc="Animating")):
        pdf_frame = mask_future(pdf_img, frame_idx, fps, max_sec)
        pdf_scaled = cv2.resize(pdf_frame, (vid_w, pdf_scaled_h))
        out_frames.append(np.vstack([vid_frame, pdf_scaled]))

    out_path = osp.join(vis_dir, f'{instance}_animated.mp4')
    write_video(out_path, out_frames, fps=fps)
    print(f"Animated video saved to: {out_path}")


if __name__ == '__main__':
    main()
