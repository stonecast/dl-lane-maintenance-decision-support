# ui/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),

    path("dashboard/", views.dashboard, name="dashboard"),
    path("upload/", views.upload_view, name="upload"),

    path("batches/", views.batch_list, name="batch_list"),
    path("batches/<int:batch_id>/", views.batch_detail, name="batch_detail"),

    # 추론 실행
    path("batches/<int:batch_id>/infer/", views.infer_batch, name="infer_batch"),

    # 필요하면 강제 로그아웃 → 로그인
    path("force-logout/", views.force_logout_then_login, name="force_logout"),
    
    # 보고서
    path("batches/<int:batch_id>/reports/", views.batch_reports, name="batch_reports"),
    path("batches/<int:batch_id>/reports/create/", views.create_batch_report, name="create_batch_report"),
]