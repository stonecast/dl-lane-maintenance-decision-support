from .models import InferenceConfig

def get_inference_config():
    cfg = InferenceConfig.objects.first()
    if cfg is None:
        cfg = InferenceConfig.objects.create(name="default")
    return cfg

# ui/utils.py
from .models import TrafficConfig

def get_traffic_config():
    """
    가장 최근에 수정된 TrafficConfig 1개 리턴.
    없으면 None.
    """
    cfg = TrafficConfig.objects.order_by("-updated_at").first()
    return cfg
