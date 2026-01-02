# ui/views.py
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import logout
from django.contrib import messages
from django.db.models import Count
from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from django.utils import timezone

import os
import json
import csv
import io
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta

from .models import UploadBatch, ImageItem, Report
from .forms import UploadBatchSelectForm
from .inference import run_inference_for_path
from django.urls import reverse
from .report_service import create_report_for_batch, build_report_view_context


# ─────────────────────────────
# 공통 유틸 / 로그인 흐름
# ─────────────────────────────
def force_logout_then_login(request):
    if request.user.is_authenticated:
        logout(request)
    return redirect("login")


def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return redirect("login")


def staff_required(view_func):
    return user_passes_test(lambda u: u.is_active and u.is_staff)(view_func)


# ─────────────────────────────
# 대시보드
# ─────────────────────────────
@login_required
def dashboard(request):
    # 상단 카드용 통계 (전체 기준)
    total_uploaded = ImageItem.objects.count()
    infer_done     = ImageItem.objects.exclude(status="WAIT").count()
    danger_count   = ImageItem.objects.filter(status="BAD").count()
    report_count   = Report.objects.count()

    recent_batches = UploadBatch.objects.order_by("-created_at")[:6]

    latest_batch = UploadBatch.objects.order_by("-created_at").first()
    points = []

    if latest_batch:
        images = latest_batch.images.exclude(lat__isnull=True, lon__isnull=True)
        for img in images:
            has_vis = img.status in ("OK", "BAD")
            vis_url = ""
            if has_vis:
                vis_url = f"/media/inference/batch_{latest_batch.id}/vis/{img.id}.jpg"

            points.append({
                "id": img.id,
                "lat": img.lat,
                "lon": img.lon,
                "status": img.status,
                "name": img.title or img.file.name,
                "original_url": img.file.url,
                "vis_url": vis_url,
                "has_vis": has_vis,
            })

    context = {
        "total_uploaded": total_uploaded,
        "infer_done": infer_done,
        "danger_count": danger_count,
        "report_count": report_count,
        "recent_batches": recent_batches,
        "latest_batch": latest_batch,
        "points_json": json.dumps(points),
    }
    return render(request, "pages/dashboard.html", context)


# ─────────────────────────────
# 배치 목록 / 상세 / 업로드 / 추론
# ─────────────────────────────
VALID_STATUS = {"OK", "BAD", "WAIT"}
VALID_SORT   = {"recent", "oldest", "congestion_high", "congestion_low"}

@login_required
def batch_list(request):
    batches = UploadBatch.objects.order_by("-created_at")

    selected = request.GET.get("selected")
    sel = UploadBatch.objects.filter(pk=selected).first() if selected else None

    images = sel.images.order_by("-uploaded_at") if sel else []

    points = [
        {
            "lat": it.lat,
            "lon": it.lon,
            "name": (it.title or it.file.name),
            "status": (it.status or ""),
        }
        for it in images
        if it.lat is not None and it.lon is not None
    ]

    return render(
        request,
        "pages/batch_list.html",
        {
            "batches": batches,
            "selected": sel,
            "points": points,
        },
    )



@login_required
def batch_detail(request, batch_id: int):
    batch = get_object_or_404(UploadBatch, pk=batch_id)

    images_qs = ImageItem.objects.filter(batch=batch)

    ok_count = images_qs.filter(status="OK").count()
    bad_count = images_qs.filter(status="BAD").count()
    wait_count = images_qs.filter(status="WAIT").count()

    status_param = request.GET.get("status", "ALL")
    status_multi = request.GET.getlist("status[]")

    active_statuses = None
    if status_multi:
        active_statuses = [s.upper() for s in status_multi if s]
    elif status_param and status_param != "ALL":
        active_statuses = [status_param.upper()]

    if active_statuses:
        images_qs = images_qs.filter(status__in=active_statuses)

    sort = request.GET.get("sort", "recent")
    if sort == "recent":
        images_qs = images_qs.order_by("-uploaded_at")
    elif sort == "oldest":
        images_qs = images_qs.order_by("uploaded_at")
    elif sort == "congestion_high":
        images_qs = images_qs.order_by("-congestion", "-uploaded_at")
    elif sort == "congestion_low":
        images_qs = images_qs.order_by("congestion", "-uploaded_at")

    points = []
    for it in images_qs:
        if it.lat is not None and it.lon is not None:
            vis_rel = f"inference/batch_{batch.id}/vis/{it.id}.jpg"
            vis_url = settings.MEDIA_URL + vis_rel

            points.append(
                {
                    "id": it.id,
                    "lat": it.lat,
                    "lon": it.lon,
                    "name": it.title or it.file.name,
                    "status": it.status,
                    "congestion": it.congestion,
                    "original_url": it.file.url,
                    "vis_url": vis_url,
                }
            )

    points_json = json.dumps(points, ensure_ascii=False)

    context = {
        "batch": batch,
        "images": images_qs,
        "ok_count": ok_count,
        "bad_count": bad_count,
        "wait_count": wait_count,
        "status_filter": status_param,
        "sort_option": sort,
        "points_json": points_json,
    }
    return render(request, "pages/batch_detail.html", context)


def _parse_metadata_file(django_file):
    if not django_file:
        return {}
    try:
        raw = django_file.read()
        text = raw.decode("utf-8-sig", errors="ignore")
    except Exception:
        return {}
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "\t" if "\t" in first_line else ","
    f = io.StringIO(text)
    reader = csv.DictReader(f, delimiter=delimiter)
    mapping = {}

    def norm_key(k: str) -> str:
        return k.strip().strip('"').strip().lower()

    fieldnames = reader.fieldnames or []
    norm_fields = {norm_key(k): k for k in fieldnames}
    fname_keys = [k for k in norm_fields if "file" in k and "name" in k] or ["filename"]
    lat_keys   = [k for k in norm_fields if "lat" in k]
    lon_keys   = [k for k in norm_fields if "lon" in k or "lng" in k]

    for row in reader:
        row_norm = {norm_key(k): v for k, v in row.items()}
        fn = None
        for k in fname_keys:
            if k in row_norm and row_norm[k]:
                fn = row_norm[k]
                break
        if not fn:
            continue

        lat_str = None
        for k in lat_keys:
            if k in row_norm and row_norm[k]:
                lat_str = row_norm[k]
                break

        lon_str = None
        for k in lon_keys:
            if k in row_norm and row_norm[k]:
                lon_str = row_norm[k]
                break

        if not lat_str or not lon_str:
            continue

        try:
            lat = float(str(lat_str).strip())
            lon = float(str(lon_str).strip())
        except ValueError:
            continue

        base = os.path.basename(fn).lower()
        mapping[base] = (lat, lon)

    return mapping


@login_required
def upload_view(request):
    if request.method == "POST":
        mode     = request.POST.get("mode", "new")
        title    = request.POST.get("title", "").strip()
        batch_id = request.POST.get("batch", "").strip()

        # 배치 선택/생성
        if mode == "existing" and batch_id:
            batch = get_object_or_404(UploadBatch, pk=batch_id)
        else:
            if not title:
                messages.error(request, "배치 이름을 입력하세요.")
                return redirect("upload")
            batch = UploadBatch.objects.create(
                title=title,
                status="READY",
                description=""
            )

        # 이미지 파일
        files = request.FILES.getlist("files")
        if not files:
            messages.error(request, "업로드할 파일이 없습니다.")
            return redirect("upload")

        # 메타데이터 CSV (선택) - 위에 정의한 _parse_metadata_file 사용
        meta_file = request.FILES.get("meta_file")
        meta_map = {}
        if meta_file:
            try:
                meta_map = _parse_metadata_file(meta_file)
            except Exception as e:
                print("[meta csv parse error]", e)
                messages.warning(request, "메타데이터 CSV 파싱 중 오류가 발생했습니다.")

        # 4) 이미지 생성 + 메타데이터 매핑
        created = 0
        for f in files:
            item = ImageItem.objects.create(
                batch=batch,
                file=f,
                title=f.name,
            )
            created += 1

            if meta_map:
                base = os.path.basename(f.name).lower()
                if base in meta_map:
                    item.lat, item.lon = meta_map[base]
                    item.save(update_fields=["lat", "lon"])

        messages.success(request, f"{created}개 파일 업로드 완료")
        # 업로드 후에는 해당 배치 상세 페이지로 이동
        return redirect("batch_detail", batch_id=batch.id)

    form = UploadBatchSelectForm()
    return render(request, "pages/upload.html", {"form": form})



@login_required
def infer_batch(request, batch_id):
    batch = get_object_or_404(UploadBatch, pk=batch_id)
    images = batch.images.all()
    out_root = os.path.join(settings.MEDIA_ROOT, f"inference/batch_{batch.id}")
    batch.status = "RUNNING"
    batch.save(update_fields=["status"])

    for img in images:
        image_path = img.file.path
        status, congestion = run_inference_for_path(image_path, out_root, img.id)
        img.status = status
        img.congestion = congestion
        img.save(update_fields=["status", "congestion"])

    batch.status = "DONE"
    batch.save(update_fields=["status"])

    messages.success(request, f"{images.count()}장에 대해 추론을 완료했습니다.")
    return redirect("batch_detail", batch_id=batch.id)

# ─────────────────────────────
# 보고서 보기
# ─────────────────────────────
@login_required
def batch_reports(request, batch_id):
    batch = get_object_or_404(UploadBatch, id=batch_id)
    report_id = request.GET.get("report_id")
    context = build_report_view_context(batch, report_id)
    return render(request, "pages/batch_report.html", context)

# ─────────────────────────────
# 보고서 생성 호출
# ─────────────────────────────
@login_required
def create_batch_report(request, batch_id):
    batch = get_object_or_404(UploadBatch, id=batch_id)

    if request.method != "POST":
        return redirect("batch_reports", batch_id=batch.id)

    if not getattr(settings, "TMAP_APP_KEY", "").strip():
        messages.error(request, "TMAP_APP_KEY가 설정되지 않았습니다. (.env 또는 환경변수/ settings.py 확인)")
        return redirect("batch_reports", batch_id=batch.id)

    try:
        report = create_report_for_batch(batch)
    except ValueError as e:
        messages.error(request, str(e))
        return redirect("batch_reports", batch_id=batch.id)
    except Exception as e:
        print("[create_batch_report] unexpected error:", e)
        messages.error(request, "보고서 생성 중 알 수 없는 오류가 발생했습니다.")
        return redirect("batch_reports", batch_id=batch.id)

    messages.success(request, "새 보고서를 생성했습니다.")
    return redirect(f"/batches/{batch.id}/reports/?report_id={report.id}")
