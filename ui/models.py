# ui/models.py
from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver


class UploadBatch(models.Model):
    title = models.CharField(max_length=200)
    status = models.CharField(
        max_length=20,
        default="READY",
        help_text="READY / RUNNING / DONE 등 상태표시용",
    )
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.id}: {self.title}"


class ImageItem(models.Model):
    batch = models.ForeignKey(
        UploadBatch,
        related_name="images",
        on_delete=models.CASCADE,
    )
    file = models.ImageField(upload_to="images/%Y/%m/%d/")
    title = models.CharField(max_length=200, blank=True, null=True)

    status = models.CharField(
        max_length=10,
        choices=[("OK", "정상"), ("BAD", "불량"), ("WAIT", "대기")],
        default="WAIT",
        blank=False,
        null=False,
    )

    congestion = models.FloatField(blank=True, null=True, default=None)

    lat = models.FloatField(blank=True, null=True)
    lon = models.FloatField(blank=True, null=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or self.file.name


@receiver(post_delete, sender=ImageItem)
def delete_file_on_imageitem_delete(sender, instance, **kwargs):
    """ImageItem 삭제 시 실제 파일도 같이 삭제."""
    if instance.file:
        instance.file.delete(save=False)


class Report(models.Model):
    """
    한 번 Tmap/CSV/분석을 돌린 결과를 그대로 JSON으로 저장하는 보고서 엔티티.
    - payload 안에 거의 모든 내용을 때려 넣고
    - 화면에서는 payload만 읽어서 렌더
    """
    batch = models.ForeignKey(
        UploadBatch,
        related_name="reports",
        on_delete=models.CASCADE,
    )
    title = models.CharField(max_length=200, default="자동 생성 보고서")

    # 분석 결과 전체 JSON
    payload = models.JSONField(default=dict, blank=True)

    # 화면/문서용 짧은 요약문
    summary = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report {self.id} for Batch {self.batch_id}"


class InferenceConfig(models.Model):
    name = models.CharField(max_length=100, default="default")
    defect_ratio_thr = models.FloatField(default=0.05)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (thr={self.defect_ratio_thr})"


class TrafficConfig(models.Model):
    """
    교통/비닝 관련 설정:
      - Tmap APP KEY
      - IC/JC 목록 CSV
      - 시간대별 교통량 CSV
    """
    name = models.CharField(max_length=100, default="default", unique=True)
    tmap_app_key = models.CharField("Tmap APP KEY", max_length=255)

    ic_csv = models.FileField(
        "IC/JC 목록 CSV",
        upload_to="config/",
        blank=True,
        null=True,
        help_text="IC/JC 좌표가 들어있는 CSV 파일"
    )
    hourly_csv = models.FileField(
        "시간대별 교통량 CSV",
        upload_to="config/",
        blank=True,
        null=True,
        help_text="영업소/집계시/교통량이 들어있는 CSV 파일"
    )
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"[TrafficConfig] {self.name}"
