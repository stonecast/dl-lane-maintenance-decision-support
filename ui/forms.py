from django import forms
from .models import UploadBatch

class UploadBatchSelectForm(forms.Form):
    MODE_CHOICES = (("new", "새 배치 만들기"), ("existing", "기존 배치에 추가"))
    mode  = forms.ChoiceField(choices=MODE_CHOICES, widget=forms.RadioSelect, initial="new")
    title = forms.CharField(label="배치 이름", max_length=200, required=False)
    batch = forms.ModelChoiceField(label="기존 배치 선택",
                                   queryset=UploadBatch.objects.none(),
                                   required=False)
    description = forms.CharField(label="설명",
                                  required=False,
                                  widget=forms.Textarea(attrs={"rows": 2})
                                  )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["batch"].queryset = UploadBatch.objects.all().order_by("-created_at")

    def clean(self):
        cleaned = super().clean()
        mode  = cleaned.get("mode")
        title = cleaned.get("title")
        batch = cleaned.get("batch")
        if mode == "new" and not title:
            self.add_error("title", "새 배치 이름을 입력하세요.")
        if mode == "existing" and not batch:
            self.add_error("batch", "추가할 기존 배치를 선택하세요.")
        return cleaned
