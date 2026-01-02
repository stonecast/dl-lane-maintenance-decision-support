from django.contrib import admin
from .models import UploadBatch, ImageItem, InferenceConfig, TrafficConfig

@admin.action(description="선택한 배치와 하위 이미지 모두 삭제")
def delete_batches_and_files(modeladmin, request, queryset):
    # CASCADE + post_delete 로 실제 파일까지 정리됨
    queryset.delete()

@admin.action(description="선택한 이미지 파일 삭제")
def delete_images_and_files(modeladmin, request, queryset):
    queryset.delete()

@admin.register(UploadBatch)
class UploadBatchAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "status", "created_at", "image_count")
    actions = [delete_batches_and_files]
    def image_count(self, obj):
        return obj.images.count()

@admin.register(ImageItem)
class ImageItemAdmin(admin.ModelAdmin):
    list_display = ("id", "file", "batch", "uploaded_at", "lat", "lon", "status")
    actions = [delete_images_and_files]

@admin.register(InferenceConfig)
class InferenceConfigAdmin(admin.ModelAdmin):
    list_display = ("name", "defect_ratio_thr", "updated_at")


@admin.register(TrafficConfig)
class TrafficConfigAdmin(admin.ModelAdmin):
    list_display = ("name", "tmap_app_key", "updated_at")
