# alert_video.py
"""Alert-specific video generation for alert-pipeline.ipynb: GT, OBB-prediction, and CLS-prediction videos.

Builds on top of pmof-code's src.visualization (results_to_frames, imageids_to_gtframes, save_video) without
modifying pmof-code; the alert border is an alert-pipeline-only concept so it's drawn locally.
"""
import cv2
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import contextmanager
from tqdm import tqdm

from src.data import imgid_to_imgpath, imgid_to_annpath, read_annotation, recordid_to_imageids
from src.visualization import results_to_frames, imageids_to_gtframes, save_video
from src.visualization import frame_visualizer as _frame_visualizer

ALERT_COLORS = {0: "tab:green", 1: "tab:pink"}  # 0: no alert, 1: alert
BORDER_THICKNESS = 20
DASH_LENGTH = 30
GAP_LENGTH = 20
GT_INSET = 0     # solid GT border sits flush with the frame edge
PRED_INSET = 35  # dashed prediction border nested inside the GT border
# pmof-code's legend figure doesn't set its own dpi, so it inherits whatever figure.dpi the notebook has set
# globally (e.g. 300 for print-quality plots); at that dpi the legend image can be taller than a video frame.
LEGEND_DPI = 100


@contextmanager
def _silence_inner_progress():
    """Suppress pmof-code's own tqdm bar while we call it once per frame (one bar per video otherwise)."""
    original_tqdm = _frame_visualizer.tqdm
    _frame_visualizer.tqdm = lambda iterable, **kwargs: iterable
    try:
        yield
    finally:
        _frame_visualizer.tqdm = original_tqdm


def _colorstr_to_rgb255(color):
    """Convert a matplotlib color string/hex to an (R, G, B) 0-255 tuple (frames here are RGB, not BGR)."""
    r, g, b = mcolors.to_rgb(color)
    return (int(r * 255), int(g * 255), int(b * 255))


def _draw_dashed_line(img_rgb, pt1, pt2, color, thickness):
    x1, y1 = pt1
    x2, y2 = pt2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    step = DASH_LENGTH + GAP_LENGTH
    for start in range(0, length, step):
        end = min(start + DASH_LENGTH, length)
        seg_start = (int(x1 + dx * start), int(y1 + dy * start))
        seg_end = (int(x1 + dx * end), int(y1 + dy * end))
        cv2.line(img_rgb, seg_start, seg_end, color, thickness)


def draw_alert_border(img_rgb, alert, dashed, thickness=BORDER_THICKNESS):
    """
    Draw a solid (GT) or dashed (prediction) rectangle border signaling alert/no-alert, inset from the frame edge.

    Parameters
    -------
    img_rgb: np.array
        image in RGB format, modified in place.

    alert: int
        0 (no alert) or 1 (alert); selects the color from ALERT_COLORS.

    dashed: bool
        if True, draw a dashed border (prediction); if False, draw a solid border (ground truth).

    Returns
    -------
    np.array
        img_rgb with the border drawn.
    """
    color = _colorstr_to_rgb255(ALERT_COLORS[alert])
    inset = PRED_INSET if dashed else GT_INSET

    h, w = img_rgb.shape[:2]
    x1, y1 = inset, inset
    x2, y2 = w - 1 - inset, h - 1 - inset

    if not dashed:
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thickness)
        return img_rgb

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    for pt1, pt2 in zip(corners[:-1], corners[1:]):
        _draw_dashed_line(img_rgb, pt1, pt2, color, thickness)
    return img_rgb


def _gt_alert_for_frame(image_id, data_base_dir):
    """Return 1 if any non-occluded person annotation has a non-'seated' action, else 0."""
    annotation_path = imgid_to_annpath(data_base_dir=data_base_dir, image_id=image_id)
    annotations = read_annotation(annotation_path=annotation_path, image_id=image_id)
    person_anns = [ann for ann in annotations if ann.category_name == "person" and not ann.occluded]
    return 1 if any(ann.action != "seated" for ann in person_anns) else 0


def _require_matching_length(name, values, expected_length):
    if len(values) != expected_length:
        raise ValueError(f"{name} has length {len(values)}, expected {expected_length}.")


def generate_gt_alert_video(record_id, data_base_dir, output_path=None, fps=10, preview=False, frame_limit=None,
                             actions=True, gt_person_only=True):
    """
    Generate a video of ground-truth frames: GT boxes + legend (pmof-code) + a solid alert/no-alert border.

    Parameters
    -------
    record_id: str
        record id, e.g. "rec26".

    data_base_dir: str or Path
        path to the dataset.

    output_path, fps, preview, frame_limit:
        forwarded to save_video.

    actions: bool (optional)
        if True, label boxes with their action instead of category name.

    gt_person_only: bool (optional)
        if True, only "person" category boxes are drawn (drops bag/clothing/suitcase).

    Returns
    -------
    str
        output_path of the saved video.
    """
    image_ids = recordid_to_imageids(record_id, data_base_dir=data_base_dir)

    frames = []
    with plt.rc_context({"figure.dpi": LEGEND_DPI}), _silence_inner_progress():
        for image_id in tqdm(image_ids, desc="Rendering GT alert video"):
            # one image_id per call: pmof-code's FrameVisualizer keeps one legend_builder per call,
            # so batching would accumulate every label ever seen into every frame's legend.
            frame = imageids_to_gtframes([image_id], data_base_dir=data_base_dir, actions=actions,
                                          gt_person_only=gt_person_only)[image_id]
            draw_alert_border(frame, _gt_alert_for_frame(image_id, data_base_dir), dashed=False)
            frames.append(frame)

    return save_video(frames, output_path=output_path, frame_limit=frame_limit, fps=fps, preview=preview)


def generate_obb_alert_video(obb_results, obb_alerts, data_base_dir, output_path=None, fps=10, preview=False,
                              frame_limit=None, display_gt_obb=False, display_gt_alert=False, gt_alerts=None, actions=True, gt_person_only=True):
    """
    Generate a video of OBB predictions: predicted boxes + legend (pmof-code) + a dashed alert border.

    Parameters
    -------
    obb_results: list
        ultralytics Results, one per frame (e.g. frame_results['obb_results'] in alert-pipeline.ipynb, flattened
        since each cached entry is itself a 1-element list).

    obb_alerts: list[int]
        per-frame predicted alert flag (0/1), same order/length as obb_results.

    data_base_dir: str or Path
        path to the dataset. Only used when display_gt=True.

    display_gt_obb: bool (optional)
        if True, additionally overlay GT OBB boxes/legend (via pmof-code).
    
    display_gt_alert: bool (optional)
        if True, additionally overlay a solid GT alert border.

    gt_alerts: list[int] (optional)
        per-frame GT alert flag (0/1), same order/length as obb_results. Required when display_gt=True.

    actions: bool (optional)
        if True, label GT boxes with their action instead of category name. Only used when display_gt=True.

    gt_person_only: bool (optional)
        if True, only "person" category GT boxes are drawn. Only used when display_gt=True.

    Returns
    -------
    str
        output_path of the saved video.
    """
    _require_matching_length("obb_alerts", obb_alerts, len(obb_results))
    if display_gt_obb or display_gt_alert:
        if gt_alerts is None:
            raise ValueError("gt_alerts is required when display_gt_obb=True or display_gt_alert=True")
        _require_matching_length("gt_alerts", gt_alerts, len(obb_results))

    frames = []
    with plt.rc_context({"figure.dpi": LEGEND_DPI}), _silence_inner_progress():
        for i, result in enumerate(tqdm(obb_results, desc="Rendering OBB alert video")):
            image_id = Path(result.path).stem
            # one result per call: see generate_gt_alert_video for why legends must not be batched.
            frame = results_to_frames([result], data_base_dir=data_base_dir, actions=actions,
                                       gt_person_only=gt_person_only, display_gt=display_gt_obb)[image_id]
            if display_gt_alert:
                draw_alert_border(frame, gt_alerts[i], dashed=False)
            draw_alert_border(frame, obb_alerts[i], dashed=True)
            frames.append(frame)

    return save_video(frames, output_path=output_path, frame_limit=frame_limit, fps=fps, preview=preview)


def generate_cls_alert_video(record_id, cls_alerts, data_base_dir, output_path=None, fps=10, preview=False,
                              frame_limit=None, display_gt_obb=False, display_gt_alert=False, gt_alerts=None, actions=True, gt_person_only=True):
    """
    Generate a video of CLS predictions: a dashed alert border only (the cls model produces no boxes).

    Parameters
    -------
    record_id: str
        record id, e.g. "rec28".

    cls_alerts: list[int]
        per-frame predicted alert flag (0/1), aligned with recordid_to_imageids(record_id).

    data_base_dir: str or Path
        path to the dataset.

    display_gt_obb: bool (optional)
        if True, overlay GT OBB boxes/legend (via pmof-code).

    display_gt_alert: bool (optional)
        if True, overlay a solid GT alert border.

    gt_alerts: list[int] (optional)
        per-frame GT alert flag (0/1), same order/length as cls_alerts. Required when display_gt=True.

    actions: bool (optional)
        if True, label GT boxes with their action instead of category name. Only used when display_gt=True.

    gt_person_only: bool (optional)
        if True, only "person" category GT boxes are drawn. Only used when display_gt=True.

    Returns
    -------
    str
        output_path of the saved video.
    """
    image_ids = recordid_to_imageids(record_id, data_base_dir=data_base_dir)
    _require_matching_length("cls_alerts", cls_alerts, len(image_ids))
    if display_gt_obb or display_gt_alert:
        if gt_alerts is None:
            raise ValueError("gt_alerts is required when display_gt_obb=True or display_gt_alert=True")
        _require_matching_length("gt_alerts", gt_alerts, len(image_ids))

    frames = []
    with plt.rc_context({"figure.dpi": LEGEND_DPI}), _silence_inner_progress():
        for i, image_id in enumerate(tqdm(image_ids, desc="Rendering CLS alert video")):
            if display_gt_obb:
                # one image_id per call: see generate_gt_alert_video for why legends must not be batched.
                frame = imageids_to_gtframes([image_id], data_base_dir=data_base_dir, actions=actions,
                                              gt_person_only=gt_person_only)[image_id]
            else:
                frame = cv2.cvtColor(cv2.imread(imgid_to_imgpath(data_base_dir=data_base_dir, image_id=image_id)),
                                      cv2.COLOR_BGR2RGB)
            if display_gt_alert:
                draw_alert_border(frame, gt_alerts[i], dashed=False)
            draw_alert_border(frame, cls_alerts[i], dashed=True)
            frames.append(frame)

    return save_video(frames, output_path=output_path, frame_limit=frame_limit, fps=fps, preview=preview)
