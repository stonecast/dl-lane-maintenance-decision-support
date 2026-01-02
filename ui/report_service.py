from __future__ import annotations
from typing import Optional

import json
import pandas as pd

from django.urls import reverse
from django.utils import timezone
from django.conf import settings

from .models import Report
from .traffic_utils import analyze_segments, rank_segments_for_now
from .utils import get_traffic_config


def build_priority_reason_rulebased(highlight_segment: dict, ranked: list[dict]) -> str:
    if not highlight_segment:
        return ""

    # 랭킹 계산
    try:
        idx = ranked.index(highlight_segment)
    except ValueError:
        key = (
            highlight_segment.get("line"),
            highlight_segment.get("from_ic"),
            highlight_segment.get("to_ic"),
            highlight_segment.get("direction"),
        )
        idx = 0
        for i, r in enumerate(ranked):
            k2 = (r.get("line"), r.get("from_ic"), r.get("to_ic"), r.get("direction"))
            if k2 == key:
                idx = i
                break

    rank = idx + 1
    total_seg = len(ranked)

    defect = highlight_segment.get("defect", 0) or 0
    total = highlight_segment.get("total", 0) or 0
    defect_ratio = (defect / total * 100.0) if total else 0.0

    traffic = highlight_segment.get("traffic") or {}
    speed = traffic.get("speed")
    congestion = traffic.get("congestion")

    has_more_defect_segment = any((r.get("defect", 0) or 0) > defect for r in ranked)

    line = highlight_segment.get("line") or ""
    from_ic = highlight_segment.get("from_ic") or ""
    to_ic = highlight_segment.get("to_ic") or ""
    direction = highlight_segment.get("direction") or ""

    parts = []
    parts.append(f"{line} {from_ic} → {to_ic} {direction} 방면에서")
    parts.append(f" 총 {total}개 이미지 중 불량 차선이 {defect}개(약 {defect_ratio:.0f}%) 탐지되었습니다.\n")

    if has_more_defect_segment and defect > 0:
        parts.append("다른 구간들에서 불량 차선이 더 많이 발견된 곳도 존재하지만,")

    if speed is not None or congestion is not None:
        cong_txt = ""
        if congestion is not None:
            cong_txt = f"현재 혼잡도 {congestion} 수준이며, "
        if speed is not None:
            parts.append(f"{cong_txt}실제 통행 속도는 약 {speed} km/h로 비교적 원활한 편입니다.\n")
        else:
            parts.append(f"{cong_txt}현재 교통 상황을 고려했을 때, 상대적으로 보수 작업이 수월한 시간대에 해당합니다.")

    parts.append(
        "\n따라서 교통 흐름에 미치는 영향을 최소화하면서도 불량 차선을 효율적으로 정비할 수 있어, "
        "해당 구간부터 보수 공사를 진행하는 것을 추천드립니다."
    )

    return " ".join(parts)


def create_report_for_batch(batch) -> Report:

    cfg = get_traffic_config()

    # 1) settings에서 키를 받는다
    tmap_key = getattr(settings, "TMAP_APP_KEY", "") or ""
    if not tmap_key:
        raise ValueError("TMAP_APP_KEY가 설정되지 않았습니다. (settings/env)")

    # 2) ic_csv는 기존 TrafficConfig를 그대로 사용(키만 settings로 이동)
    if cfg is None or not cfg.ic_csv:
        raise ValueError("IC/JC 목록 CSV가 설정되지 않았습니다. (TrafficConfig.ic_csv)")

    # 3) 숫자 요약
    all_images = batch.images.all()
    total_images = all_images.count()
    defect_images = all_images.filter(status="BAD").count()
    ok_images = all_images.filter(status="OK").count()
    total_inferred = all_images.exclude(status="WAIT").count()

    # 4) 위경도 있는 이미지로 points 생성
    images = all_images.filter(lat__isnull=False, lon__isnull=False)
    if not images.exists():
        raise ValueError("이 배치에는 위도/경도 정보가 있는 이미지가 없습니다.")

    points = []
    for it in images:
        if it.status == "BAD":
            st = "defect"
        elif it.status == "OK":
            st = "normal"
        else:
            st = "unknown"
        points.append((float(it.lat), float(it.lon), st, it.id))

    # 5) ic csv 로드
    try:
        ic_df = pd.read_csv(cfg.ic_csv.path, encoding="cp949")
    except Exception as e:
        raise ValueError(f"IC CSV 로딩 중 오류: {e}")

    # 6) 구간 분석/랭킹
    try:
        segment_report = analyze_segments(points, ic_df, tmap_key)
    except Exception as e:
        raise ValueError(f"구간 분석 중 오류: {e}")

    if not segment_report:
        raise ValueError("분석 가능한 구간이 없습니다.")

    ranked = rank_segments_for_now(segment_report)

    # 7) highlight 선택: 불량>0 우선
    highlight_segment = None
    for row in ranked:
        if (row.get("defect") or 0) > 0:
            highlight_segment = row
            break
    if highlight_segment is None and ranked:
        highlight_segment = ranked[0]

    # 8) summary 텍스트 구성
    now = timezone.now()
    title = now.strftime("%Y-%m-%d %H:%M 기준 유지보수 추천 보고서")

    if highlight_segment:
        line = highlight_segment.get("line", "")
        from_ic = highlight_segment.get("from_ic", "")
        to_ic = highlight_segment.get("to_ic", "")
        direction = highlight_segment.get("direction", "")
        worst_defects = highlight_segment.get("defect", 0) or 0
        main_sentence = (
            f"해당 배치에서는 {line} {from_ic} ~ {to_ic} {direction} 방면에서 "
            f"가장 많은 불량 차선({worst_defects}개)이 발생했습니다."
        )
    else:
        main_sentence = "분석 가능한 구간이 충분하지 않아 주요 불량 구간을 특정하지 못했습니다."

    summary = (
        f"총 {total_images}개 이미지에 대해 추론을 진행하였으며, "
        f"이 중 불량 차선 {defect_images}개, 정상 차선 {ok_images}개가 확인되었습니다.\n"
        f"{main_sentence}\n\n"
        f"실시간 교통 혼잡도와 불량 비율을 함께 고려하여, 보수 우선순위를 산정했습니다."
    )

    # 9) 규칙 기반 추천 사유
    now_reason = ""
    if highlight_segment:
        now_reason = build_priority_reason_rulebased(highlight_segment, ranked)

    payload = {
        "generated_at": now.isoformat(),
        "total_images": total_images,
        "defect_images": defect_images,
        "ok_images": ok_images,
        "total_inferred": total_inferred,
        "segments": segment_report,
        "now_ranking": ranked,
        "highlight_segment": highlight_segment,
        "now_reason": now_reason,
    }

    report = Report.objects.create(
        batch=batch,
        title=title,
        summary=summary,
        payload=payload,
    )
    return report

def build_report_view_context(batch, report_id: Optional[str] = None) -> dict:
    """
    보고서 '보기' 페이지에서 사용할 context를 구성한다.
    - view에서는 batch만 가져오고, 본 함수 결과를 render에 넘긴다.
    - 템플릿(pages/batch_report.html)이 기대하는 키들을 그대로 반환한다.
    """
    reports = Report.objects.filter(batch=batch).order_by("-created_at")

    # 보고서가 하나도 없으면 템플릿이 깨지지 않게 빈 context 반환
    if not reports.exists():
        return {
            "batch": batch,
            "reports": [],
            "selected_report": None,
            "summary": None,
            "segments": [],
            "segments_with_defect_count": 0,
            "top_now": None,
            "top_now_previews": [],
            "detail_segments": [],
            "segment_points_json": "[]",
            "top_now_reason": "",
        }

    # 선택된 보고서 결정
    if report_id:
        selected = Report.objects.filter(id=report_id, batch=batch).first()
        if selected is None:
            selected = reports.first()
    else:
        selected = reports.first()

    # payload 안전 파싱
    try:
        payload = selected.payload or {}
    except Exception:
        payload = {}

    segments = payload.get("segments") or []
    now_ranking_raw = payload.get("now_ranking") or []

    # 구간별 현황: 불량 수량 기준 내림차순 정렬
    segments = sorted(segments, key=lambda s: s.get("defect", 0) or 0, reverse=True)

    # 추천 사유(LLM 또는 fallback)
    top_now_reason = payload.get("now_reason") or ""

    images_qs = batch.images.all()
    image_map = {img.id: img for img in images_qs}

    def make_preview(img_id):
        img = image_map.get(img_id)
        if not img:
            return None
        vis_url = f"/media/inference/batch_{batch.id}/vis/{img.id}.jpg"
        detail_url = reverse("batch_detail", args=[batch.id]) + f"?image_id={img.id}"
        return {
            "id": img.id,
            "title": img.title or img.file.name,
            "vis_url": vis_url,
            "detail_url": detail_url,
        }

    # ─ 1) 전체 요약 숫자 ─
    total_images = payload.get("total_images")
    if total_images is None:
        total_images = images_qs.count()

    defect_images = payload.get("defect_images")
    if defect_images is None:
        defect_images = images_qs.filter(status="BAD").count()

    ok_images = payload.get("ok_images")
    if ok_images is None:
        ok_images = images_qs.filter(status="OK").count()

    total_inferred = payload.get("total_inferred")
    if total_inferred is None:
        total_inferred = images_qs.exclude(status__isnull=True).exclude(status="WAIT").count()

    segment_count = len(segments)
    segments_with_defect_count = sum(1 for s in segments if (s.get("defect", 0) or 0) > 0)

    summary = {
        "generated_at": payload.get("generated_at"),
        "total_images": total_images,
        "total_inferred": total_inferred,
        "defect_images": defect_images,
        "ok_images": ok_images,
        "segment_count": segment_count,
        "segments_with_defect_count": segments_with_defect_count,
    }

    # ─ 2) 현재 우선순위 구간 (불량 0인 구간은 제외) ─
    now_ranking = [r for r in now_ranking_raw if (r.get("defect", 0) or 0) > 0]
    top_now = now_ranking[0] if now_ranking else None
    top_now_previews = []

    if top_now:
        img_ids = top_now.get("defect_image_ids") or top_now.get("image_ids") or []
        previews = [make_preview(i) for i in img_ids]
        top_now_previews = [p for p in previews if p][:3]

    # ─ 3) 상세 이미지 섹션: "불량이 있는 모든 구간" + 불량 포함 이미지 4장 ─
    detail_segments = []
    for row in segments:
        defect = row.get("defect", 0) or 0
        if defect <= 0:
            continue

        img_ids = row.get("defect_image_ids") or row.get("image_ids") or []
        previews = [make_preview(i) for i in img_ids]
        previews = [p for p in previews if p][:4]

        total = row.get("total", 0) or 0
        defect_ratio = row.get("defect_ratio")
        if defect_ratio is None:
            defect_ratio = (defect / total * 100.0) if total > 0 else 0.0

        congestion_score = row.get("congestion_score")
        if congestion_score is None:
            congestion_score = 0.5

        info = dict(row)
        info["defect_ratio"] = defect_ratio
        info["congestion_score"] = congestion_score

        detail_segments.append({"info": info, "previews": previews})

    # ─ 4) 지도용 포인트: top_now 구간에 속한 이미지들만 ─
    segment_points = []
    if top_now:
        img_ids_for_top = top_now.get("defect_image_ids") or top_now.get("image_ids") or []
        imgs_for_top = images_qs.filter(
            id__in=img_ids_for_top,
            lat__isnull=False,
            lon__isnull=False,
        )
        for img in imgs_for_top:
            segment_points.append({
                "id": img.id,
                "lat": img.lat,
                "lon": img.lon,
                "name": img.title or img.file.name,
                "status": img.status,
            })

    segment_points_json = json.dumps(segment_points, ensure_ascii=False)

    # (선택) 디버그 로그는 DEBUG에서만
    if getattr(settings, "DEBUG", False):
        try:
            print("[DEBUG] batch_reports:", "selected_id=", selected.id, "payload keys=", list(payload.keys()))
        except Exception:
            pass

    return {
        "batch": batch,
        "reports": reports,
        "selected_report": selected,
        "summary": summary,
        "segments": segments,
        "segments_with_defect_count": segments_with_defect_count,
        "top_now": top_now,
        "top_now_previews": top_now_previews,
        "detail_segments": detail_segments,
        "segment_points_json": segment_points_json,
        "top_now_reason": top_now_reason,
    }
