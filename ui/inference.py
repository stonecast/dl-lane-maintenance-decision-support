# ui/inference.py
import os
from pathlib import Path
import json

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

from .utils import get_inference_config  

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


# =========================================================
# 0. 2-Stage 설정 
# =========================================================
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔵 Lane Segmentation UNet (num_classes=2)
_LANE_MODEL_WEIGHTS = r"C:\Users\LWS\Documents\education\University\Tech_University_of_Korea\high_way_image_segmentation\web_ui_making\lane_detector\config\model\lane_model.pth"

# 🔴 Defect Segmentation UNet (num_classes=7, defect class idx=2)
_DEFECT_MODEL_WEIGHTS = r"C:\Users\LWS\Documents\education\University\Tech_University_of_Korea\high_way_image_segmentation\web_ui_making\lane_detector\config\model\defect_model.pth"

_IMG_SIZE         = 512   # 모델 입력 크기
_LANE_CLASS_ID    = 1     # lane 모델에서 차선 클래스 인덱스
_DEFECT_CLASS_ID  = 1     # defect 모델에서 '결함' 클래스 인덱스 (⚠️ important)

# 🔍 원하면 True로 바꿔서 출력 확인
DEBUG_INFER = False

_lane_model = None
_defect_model = None


# =========================================================
# 1. 모델 정의 (U-Net)
# =========================================================
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=32):
        super().__init__()
        self.down1 = double_conv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = double_conv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = double_conv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = double_conv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = double_conv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = double_conv(base * 2, base)

        self.outc = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        u4 = self.up4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        logits = self.outc(u1)  # (B, C, H, W)
        return logits
    

# ===================================================================
# 가중치 로드
# ===================================================================
def load_model_weights(model, ckpt_path, device="cuda", strict=True):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        if k.startswith("model."):
            k = k.replace("model.", "")
        new_state[k] = v

    model.load_state_dict(new_state, strict=strict)
    return model

# 차선, 결함 모델 가중치 로드 따로 진행
def get_two_stage_models():
    global _lane_model, _defect_model

    if (_lane_model is None) or (_defect_model is None):
        # 🔵 Lane: num_classes = 2
        lane = UNet(in_channels=1, num_classes=2, base=32).to(_DEVICE)
        load_model_weights(lane, _LANE_MODEL_WEIGHTS, device=_DEVICE, strict=False)
        lane.eval()

        # 🔴 Defect: num_classes = 7 실수로 클래스 
        defect = UNet(in_channels=1, num_classes=7, base=32).to(_DEVICE)
        load_model_weights(defect, _DEFECT_MODEL_WEIGHTS, device=_DEVICE, strict=False)
        defect.eval()

        _lane_model = lane
        _defect_model = defect

        print("[inference] Lane UNet 로드 완료:", _LANE_MODEL_WEIGHTS)
        print("[inference] Defect UNet 로드 완료:", _DEFECT_MODEL_WEIGHTS)

    return _lane_model, _defect_model


# =========================================================
# 3. 2-Stage 추론: lane + defect → 0/1/2 마스크
# =========================================================
@torch.no_grad()
def _predict_mask_from_model(model: nn.Module, img_pil: Image.Image, size: int) -> np.ndarray:
    img = img_pil.convert("L").resize((size, size), Image.BILINEAR)
    x = TF.to_tensor(img).unsqueeze(0).to(_DEVICE)  # (1,1,H,W)
    logits = model(x)
    pred = torch.argmax(logits, dim=1)[0]
    return pred.cpu().numpy().astype(np.uint8)


def _combine_lane_and_defect_masks(
    lane_mask_u8: np.ndarray,
    defect_mask_u8: np.ndarray,
    lane_class_idx: int = 1,
    defect_class_idx: int = 2,
) -> np.ndarray:
    """
    lane, defect 마스크 → 0/1/2 혼합 마스크:
      0: 배경
      1: 차선
      2: 차선 위 결함 픽셀
    """
    assert lane_mask_u8.shape == defect_mask_u8.shape, "lane/defect 마스크 크기 다름"

    lane_bin   = (lane_mask_u8 == lane_class_idx)
    defect_bin = (defect_mask_u8 == defect_class_idx)
    defect_on_lane = lane_bin & defect_bin

    merged = np.zeros_like(lane_mask_u8, dtype=np.uint8)
    merged[lane_bin] = 1
    merged[defect_on_lane] = 2
    return merged


def _infer_mask_for_path(image_path: str, size: int = _IMG_SIZE) -> np.ndarray:
    """
    ⚙ 기존: 단일 모델 → 0/1/2 마스크
    🔁 변경: 2-Stage (lane + defect) → merged 0/1/2 마스크
      0: 배경
      1: 차선
      2: 차선 위 결함
    """
    lane_model, defect_model = get_two_stage_models()

    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size

    # 1) 두 모델 각각 추론 (512x512)
    lane_mask_small   = _predict_mask_from_model(lane_model,   img, size=size)
    defect_mask_small = _predict_mask_from_model(defect_model, img, size=size)

    # 🔍 필요하면 클래스 분포 확인 (콘솔 출력)
    if DEBUG_INFER:
        uniq_lane, cnt_lane = np.unique(lane_mask_small, return_counts=True)
        uniq_def,  cnt_def  = np.unique(defect_mask_small, return_counts=True)
        print(f"[DEBUG] lane_mask unique:   {list(zip(uniq_lane.tolist(), cnt_lane.tolist()))}")
        print(f"[DEBUG] defect_mask unique: {list(zip(uniq_def.tolist(),  cnt_def.tolist()))}")

    # 2) lane + defect 결합 → merged (0/1/2)
    merged_small = _combine_lane_and_defect_masks(
        lane_mask_small,
        defect_mask_small,
        lane_class_idx=_LANE_CLASS_ID,
        defect_class_idx=_DEFECT_CLASS_ID,
    )

    if DEBUG_INFER:
        uniq_m, cnt_m = np.unique(merged_small, return_counts=True)
        print(f"[DEBUG] merged_mask unique: {list(zip(uniq_m.tolist(), cnt_m.tolist()))}")

    # 3) 원본 크기로 리사이즈 (최근접 보간, class index 보존)
    if (orig_w, orig_h) != (size, size):
        merged_img = Image.fromarray(merged_small.astype(np.uint8), mode="L")
        merged_img = merged_img.resize((orig_w, orig_h), Image.NEAREST)
        merged_u8 = np.array(merged_img, dtype=np.uint8)
    else:
        merged_u8 = merged_small.astype(np.uint8)

    return merged_u8  # 0:배경, 1:차선, 2:차선 위 결함


# =========================================================
# 4. lane 후처리 (세로 연결 + lane JSON) - 기존 그대로
# =========================================================
def _bbox(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _polygons_from_mask(mask_bool: np.ndarray, eps=2.0, min_area=50.0):
    if not _HAS_CV2:
        x, y, w, h = _bbox(mask_bool)
        return [[[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]] if w > 0 and h > 0 else []
    m = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        if eps > 0:
            cnt = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
        poly = cnt.reshape(-1, 2)
        polys.append([[int(x), int(y)] for (x, y) in poly])
    return polys


def _horiz_tolerance(bin_u8: np.ndarray, half_w: int) -> np.ndarray:
    if half_w <= 0:
        return (bin_u8 > 0).astype(np.uint8)
    if _HAS_CV2:
        kw = int(2 * half_w + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        return (cv2.dilate(bin_u8.astype(np.uint8), kernel, iterations=1) > 0).astype(np.uint8)
    H, W = bin_u8.shape
    out = bin_u8.astype(np.uint8).copy()
    for dx in range(1, half_w + 1):
        out[:, dx:] |= bin_u8[:, :-dx]
        out[:, :-dx] |= bin_u8[:, dx:]
    return (out > 0).astype(np.uint8)


def _fill_vertical_gaps(lane_bin: np.ndarray, link_px: int, min_bridge_w: int) -> np.ndarray:
    H, W = lane_bin.shape
    band = _horiz_tolerance(lane_bin, half_w=max(0, int(min_bridge_w // 2))).astype(np.uint8)

    BIG = H + 5
    dist_up = np.full((H, W), BIG, dtype=np.int32)
    last = np.full(W, -BIG, dtype=np.int32)
    for r in range(H):
        on = band[r, :] > 0
        last[on] = r
        dist_up[r, :] = r - last

    dist_dn = np.full((H, W), BIG, dtype=np.int32)
    last = np.full(W, H + BIG, dtype=np.int32)
    for r in range(H - 1, -1, -1):
        on = band[r, :] > 0
        last[on] = r
        dist_dn[r, :] = last - r

    cond_gap = (band == 0) & (dist_up <= link_px) & (dist_dn <= link_px)
    linked = (lane_bin > 0).astype(np.uint8)
    linked[cond_gap] = 1
    return linked


def process_mask_vertical(
    mask_u8: np.ndarray,
    *,
    normal_idx=1, defect_idx=2,
    link_px=30, min_bridge_w=6,
    min_lane_area=200,
    defect_ratio_thr=0.05,
    poly_simplify_eps=2.0, poly_min_area=50.0,
    export_bbox_only=False,
):
    """
    0/1/2 단일채널 마스크 → 세로 연결 → lane JSON
      0: 배경
      1: 정상 차선
      2: 차선 위 결함 픽셀
    """
    H, W = mask_u8.shape
    lane_bin = ((mask_u8 == normal_idx) | (mask_u8 == defect_idx)).astype(np.uint8)
    if not lane_bin.any():
        return {
            "size": {"width": W, "height": H},
            "lanes": [],
            "params": {
                "normal_idx": normal_idx,
                "defect_idx": defect_idx,
                "link_px": link_px,
                "min_bridge_w": min_bridge_w,
                "min_lane_area": min_lane_area,
                "defect_ratio_thr": defect_ratio_thr,
                "poly_simplify_eps": poly_simplify_eps,
                "poly_min_area": poly_min_area,
                "export_bbox_only": export_bbox_only,
            },
        }

    lane_linked = _fill_vertical_gaps(lane_bin, link_px=link_px, min_bridge_w=min_bridge_w)

    if _HAS_CV2:
        num_lbl, lbl_map = cv2.connectedComponents(lane_linked, connectivity=8)
    else:
        num_lbl, lbl_map = 2, (lane_linked > 0).astype(np.int32)

    lanes = []
    lid = 0
    for cc in range(1, num_lbl):
        region = (lbl_map == cc)

        area = int(((lane_bin > 0) & region).sum())
        if area < min_lane_area:
            continue
        defect_px = int(((mask_u8 == defect_idx) & region).sum())
        ratio = defect_px / max(area, 1)

        x, y, w, h = _bbox(region)
        status = "defect" if ratio >= defect_ratio_thr else "normal"
        if export_bbox_only or not _HAS_CV2:
            polys = [[[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]]
        else:
            polys = _polygons_from_mask(region, eps=poly_simplify_eps, min_area=poly_min_area)

        lid += 1
        lanes.append({
            "lane_id": lid,
            "status": status,
            "defect_ratio": float(ratio),
            "pixel_count": int(area),
            "bbox": [int(x), int(y), int(w), int(h)],
            "polygons": polys,
        })

    return {
        "size": {"width": W, "height": H},
        "lanes": lanes,
        "params": {
            "normal_idx": normal_idx,
            "defect_idx": defect_idx,
            "link_px": link_px,
            "min_bridge_w": min_bridge_w,
            "min_lane_area": min_lane_area,
            "defect_ratio_thr": defect_ratio_thr,
            "poly_simplify_eps": poly_simplify_eps,
            "poly_min_area": poly_min_area,
            "export_bbox_only": export_bbox_only,
        },
    }


def _load_lanejson(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    W = int(meta["size"]["width"])
    H = int(meta["size"]["height"])
    lanes = meta.get("lanes", [])
    return lanes, (W, H)


def _coerce_pts_any(poly, W, H):
    out = []
    if len(poly) == 1 and isinstance(poly[0], (list, tuple)):
        poly = poly[0]
    for p in poly:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        x, y = p[0], p[1]
        try:
            x = float(x); y = float(y)
        except:
            continue
        if np.isnan(x) or np.isnan(y):
            continue
        xi = int(max(0, min(W - 1, round(x))))
        yi = int(max(0, min(H - 1, round(y))))
        out.append((xi, yi))
    return out


def _edge_from_binary(bin_u8, width=2):
    if width < 1:
        width = 1
    img = Image.fromarray(bin_u8, mode="L")
    eroded = img
    for _ in range(width):
        eroded = eroded.filter(ImageFilter.MinFilter(3))
    edge = ImageChops.difference(img, eroded)
    return np.array(edge, dtype=np.uint8)


def visualize_lanejson(
    json_path: str,
    *,
    base_image_path: str = None,
    pred_png_path: str = None,
    normal_rgb=(0, 255, 0),
    defect_rgb=(255, 0, 0),
    lane_fill_alpha=80,
    lane_edge_alpha=180,
    defect_fill_alpha=160,     # 🔴 결함 영역 채움용
    defect_outline_alpha=230,  # 테두리
    defect_outline_width=2,
):
    lanes, (W, H) = _load_lanejson(json_path)

    if base_image_path and os.path.isfile(base_image_path):
        base = Image.open(base_image_path).convert("RGB").resize((W, H), Image.BILINEAR)
    else:
        base = Image.new("RGB", (W, H), (40, 40, 40))
    out = base.convert("RGBA")

    # 1) lane 상태(정상/불량) 폴리곤
    lane_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lane_layer)
    for lane in lanes:
        status = lane.get("status", "normal")
        polys = lane.get("polygons", [])
        rgb = defect_rgb if status == "defect" else normal_rgb
        fill = (*rgb, lane_fill_alpha)
        edge = (*rgb, lane_edge_alpha)
        for poly in polys:
            pts = _coerce_pts_any(poly, W, H)
            if len(pts) >= 3:
                d.polygon(pts, fill=fill)
                d.line(pts + [pts[0]], fill=edge, width=2)
    out = Image.alpha_composite(out, lane_layer)

    # 2) 🔴 결함 픽셀(class==2)을 별도로 빨간색으로 덮기 (lane status랑 무관)
    if pred_png_path and os.path.isfile(pred_png_path):
        pm = np.array(Image.open(pred_png_path).resize((W, H), Image.NEAREST), dtype=np.uint8)

        # (1) 결함 영역 전체를 반투명 빨간색으로 채우기
        defect_bin = (pm == 2).astype(np.uint8) * 255  # 0 or 255
        defect_mask_img = Image.fromarray(defect_bin, mode="L")
        defect_fill_layer = Image.new("RGBA", (W, H), (*defect_rgb, defect_fill_alpha))

        out = Image.alpha_composite(
            out,
            Image.composite(
                defect_fill_layer,
                Image.new("RGBA", (W, H), (0, 0, 0, 0)),
                defect_mask_img,
            ),
        )

        # (2) 테두리도 유지
        edge_mask = _edge_from_binary(defect_bin, width=defect_outline_width)
        edge_color = Image.new("RGBA", (W, H), (*defect_rgb, defect_outline_alpha))
        edge_mask_img = Image.fromarray(edge_mask, mode="L")
        out = Image.alpha_composite(
            out,
            Image.composite(
                edge_color,
                Image.new("RGBA", (W, H), (0, 0, 0, 0)),
                edge_mask_img,
            ),
        )

    return out.convert("RGB")


# =========================================================
# 5. Django에서 호출할 함수
# =========================================================
def run_inference_for_path(image_path: str, out_root: str, image_id: int):
    """
    Django view에서 사용하는 진입점.
    - image_path: 원본 이미지 경로
    - out_root : 배치별 root (예: media/inference/batch_{id})
    - image_id : DB 상 ImageItem pk
    리턴:
      status: "OK" / "BAD" / "WAIT"
      congestion: (현재는 None 유지)
    """
    cfg = get_inference_config()

    out_root = Path(out_root)
    pred_dir = out_root / "pred_masks"
    json_dir = out_root / "lanejson"
    vis_dir = out_root / "vis"
    for d in (pred_dir, json_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) 2-Stage 마스크 추론 (lane + defect → merged 0/1/2)
    mask_u8 = _infer_mask_for_path(image_path, size=_IMG_SIZE)

    # 2) lane 분석 (여기서 cfg.defect_ratio_thr 사용)
    meta = process_mask_vertical(
        mask_u8,
        normal_idx=1,
        defect_idx=2,
        link_px=30,
        min_bridge_w=6,
        min_lane_area=200,
        defect_ratio_thr=cfg.defect_ratio_thr,
        poly_simplify_eps=2.0,
        poly_min_area=50.0,
        export_bbox_only=False,
    )

    lanes = meta.get("lanes", [])
    if not lanes:
        status = "WAIT"
    else:
        has_defect = any(l.get("status") == "defect" for l in lanes)
        status = "BAD" if has_defect else "OK"

    # 3) 파일 저장 (경로/형식은 기존 그대로 유지)
    pred_path = pred_dir / f"{image_id}.png"
    Image.fromarray(mask_u8.astype(np.uint8), mode="L").save(pred_path)

    json_path = json_dir / f"{image_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    vis_img = visualize_lanejson(
        json_path=str(json_path),
        base_image_path=image_path,
        pred_png_path=str(pred_path),
    )
    vis_path = vis_dir / f"{image_id}.jpg"
    vis_img.save(vis_path, quality=95)

    congestion = None
    return status, congestion
