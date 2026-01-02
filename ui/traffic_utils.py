# ui/traffic_utils.py

import math
import numpy as np
import pandas as pd
import requests
import time


# =========================
# 0. IC CSV 불러오기 (국토부제공)
# =========================
def normalize_ic_df(ic_df: pd.DataFrame) -> pd.DataFrame:
    df = ic_df.copy()
    df.columns = [c.strip() for c in df.columns]

    print("[traffic] normalize_ic_df: raw columns =", list(df.columns))

    cols = df.columns

    lon_col = None
    lat_col = None
    code_col = None
    name_col = None
    route_name_col = None

    for c in cols:
        # 코드
        if code_col is None and ("IC/JC코드" in c or "IC코드" in c or ("코드" in c and "노선" not in c)):
            code_col = c
        # 이름
        if name_col is None and ("IC/JC명" in c or "IC명" in c or "ic_name" in c.lower()):
            name_col = c
        # 노선명
        if route_name_col is None and ("노선명" in c or "route_name" in c.lower()):
            route_name_col = c
        # 경도(X)
        if lon_col is None and (c in ["X좌표값", "경도"] or ("x" in c.lower() and "좌표" in c) or c.lower() == "lon"):
            lon_col = c
        # 위도(Y)
        if lat_col is None and (c in ["Y좌표값", "위도"] or ("y" in c.lower() and "좌표" in c) or c.lower() == "lat"):
            lat_col = c

    col_map = {}
    if code_col:
        col_map[code_col] = "ic_code"
    if name_col:
        col_map[name_col] = "ic_name"
    if route_name_col:
        col_map[route_name_col] = "route_name"
    if lon_col:
        col_map[lon_col] = "lon"
    if lat_col:
        col_map[lat_col] = "lat"

    df = df.rename(columns=col_map)

    print("[traffic] normalize_ic_df: renamed columns =", list(df.columns))

    if "ic_code" not in df.columns:
        raise ValueError(f"IC DF에 ic_code 컬럼이 없음. 현재 컬럼: {list(df.columns)}")

    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError(f"IC DF에 lon/lat 컬럼이 없음. 현재 컬럼: {list(df.columns)}")

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"])

    print("[traffic] normalize_ic_df: head=\n", df.head())
    return df


# =========================
# 1. 기본 유틸
# =========================
def find_nearest_ic(lat: float, lon: float, ic_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    df = ic_df.copy()
    dx = df["lon"] - lon
    dy = df["lat"] - lat
    df["dist"] = np.sqrt(dx ** 2 + dy ** 2)
    return df.sort_values("dist").head(top_k)


def last3_int(code) -> int:
    return int(str(code)[-3:])


def euclidean_dist(lat1, lon1, lat2, lon2) -> float:
    return math.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


# =========================
# 2. 방향 판별 + 노선/IC 구간 추론
# =========================
def infer_direction_from_coord(
    lat: float,
    lon: float,
    ic_df: pd.DataFrame,
    app_key: str,
    radius: int = 200,
    opt: int = 0,
    vehicle_type: int = 0,
    top_k: int = 5,
):
    # ─ 1) Tmap nearToRoad 호출 ─
    url = "https://apis.openapi.sk.com/tmap/road/nearToRoad"
    params = {
        "version": 1,
        "lat": lat,
        "lon": lon,
        "opt": opt,
        "vehicleType": vehicle_type,
        "radius": radius,
    }
    headers = {
        "Accept": "application/json",
        "appKey": app_key,
    }

    try:
        time.sleep(0.2)  # 너무 빨리 요청하면 에러나서 임시로 이렇게 해둠
        res = requests.get(url, params=params, headers=headers, timeout=5)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print("[traffic] nearToRoad error:", e)
        return ["판별불가", None, None, None]

    result_data = data.get("resultData")
    if not result_data:
        return ["판별불가", None, None, None]

    link_points = result_data.get("linkPoints")
    if not link_points or len(link_points) < 2:
        return ["판별불가", None, None, None]

    # 도로 시작/끝 좌표 (Tmap이 준 방향)
    start_loc = link_points[0]["location"]
    end_loc = link_points[-1]["location"]

    start_lat = float(start_loc["latitude"])
    start_lon = float(start_loc["longitude"])
    end_lat = float(end_loc["latitude"])
    end_lon = float(end_loc["longitude"])

    # ─ 2) 시작/끝 근처 IC 후보 찾기 ─
    start_near = find_nearest_ic(start_lat, start_lon, ic_df, top_k=top_k)
    end_near = find_nearest_ic(end_lat, end_lon, ic_df, top_k=top_k)

    if start_near.empty or end_near.empty:
        return ["판별불가", None, None, None]

    def get_route_name(row):
        if "route_name" in row.index and pd.notna(row["route_name"]):
            return row["route_name"]
        if "노선명" in row.index and pd.notna(row["노선명"]):
            return row["노선명"]
        return None

    def get_ic_name(row):
        if "ic_name" in row.index and pd.notna(row["ic_name"]):
            return row["ic_name"]
        if "IC/JC명" in row.index and pd.notna(row["IC/JC명"]):
            return row["IC/JC명"]
        return None

    # ─ 3) start_near × end_near 조합에서 "같은 노선 + 다른 IC 이름" 쌍 찾기 ─
    best_pair = None
    best_cost = None

    # 너무 많이 돌면 느려지니까 앞쪽 몇 개만
    start_candidates = start_near.head(min(top_k, 10))
    end_candidates = end_near.head(min(top_k, 10))

    for _, s in start_candidates.iterrows():
        rs = get_route_name(s)
        ns = get_ic_name(s)

        for _, e in end_candidates.iterrows():
            re = get_route_name(e)
            ne = get_ic_name(e)

            # 노선명이 둘 다 있고 서로 다르면 스킵 (가능하면 같은 노선만 사용)
            if rs is not None and re is not None and rs != re:
                continue

            # 같은 IC 이름(예: 판교JCT → 판교JCT)인 경우는 구간으로 안 씀
            if ns is not None and ne is not None and ns == ne:
                continue

            # 후보가 되면 거리 기반으로 가장 "자연스러운" 쌍 고르기
            cost = (
                euclidean_dist(start_lat, start_lon, s["lat"], s["lon"])
                + euclidean_dist(end_lat, end_lon, e["lat"], e["lon"])
            )

            if best_pair is None or cost < best_cost:
                best_pair = (s, e)
                best_cost = cost

    # 적절한 쌍을 못 찾으면 판별불가
    if best_pair is None:
        return ["판별불가", None, None, None]

    start_ic, end_ic = best_pair

    line_name = get_route_name(start_ic) or get_route_name(end_ic)
    start_ic_name = get_ic_name(start_ic)
    end_ic_name = get_ic_name(end_ic)

    # ─ 4) 방향(상행/하행) 계산: IC 코드 뒤 3자리 기준 ─
    try:
        s_suffix = last3_int(start_ic["ic_code"])
        e_suffix = last3_int(end_ic["ic_code"])
    except Exception:
        direction = "방향미상"
    else:
        if e_suffix > s_suffix:
            direction = "하행"
        elif e_suffix < s_suffix:
            direction = "상행"
        else:
            direction = "방향미상"

    return [direction, line_name, start_ic_name, end_ic_name]



# =========================
# 3. 여러 좌표 → 구간 비닝
# =========================
def bin_segments(points, ic_df, app_key,
                 radius=200, opt=0, vehicle_type=0, top_k=5):
    bins = {}

    for p in points:
        if p is Ellipsis:
            continue

        lat = lon = status = img_id = None

        # 튜플 / 리스트
        if isinstance(p, (list, tuple)):
            if len(p) == 2:
                lat, lon = p
                status = "unknown"
            elif len(p) >= 3:
                lat, lon, status = p[0], p[1], p[2]
                img_id = p[3] if len(p) >= 4 else None

        # dict
        elif isinstance(p, dict):
            lat = p.get("lat") or p.get("latitude")
            lon = p.get("lon") or p.get("longitude")
            status = p.get("status", "unknown")
            img_id = p.get("image_id") or p.get("id")

        if lat is None or lon is None:
            continue

        direction, line_name, start_ic_name, end_ic_name = infer_direction_from_coord(
            float(lat), float(lon), ic_df, app_key,
            radius=radius, opt=opt, vehicle_type=vehicle_type, top_k=top_k
        )

        if (
            direction == "판별불가"
            or line_name is None
            or start_ic_name is None
            or end_ic_name is None
        ):
            continue

        key = (line_name, start_ic_name, end_ic_name, direction)

        if key not in bins:
            bins[key] = {
                "normal": 0,
                "defect": 0,
                "total": 0,
                "sample_coord": (float(lat), float(lon)),
                "image_ids": [],
                "defect_image_ids": [],
            }

        # 상태 분류
        s = str(status).upper()
        if s in ["NORMAL", "정상", "0", "OK"]:
            bins[key]["normal"] += 1
        else:
            bins[key]["defect"] += 1
            if img_id is not None:
                bins[key]["defect_image_ids"].append(img_id)

        bins[key]["total"] += 1
        if img_id is not None:
            bins[key]["image_ids"].append(img_id)

    print("[traffic] bin_segments: bins keys =", list(bins.keys()))
    return bins



# =========================
# 4. 실시간(AROUND) 교통 조회
# =========================
def get_traffic_for_segment(
    sample_lat,
    sample_lon,
    app_key,
    want_direction=None,      # "상행" or "하행" or None
    want_road_type="001",     # 고속도로 코드 (TMAP 기준 001)
):
    """
    want_direction: "상행" 또는 "하행" (없으면 방향 필터 없이)
      - 상행 → direction == 0
      - 하행 → direction == 1

    want_road_type: "001" 이면 고속도로만 사용
    """

    url = "https://apis.openapi.sk.com/tmap/traffic"
    params = {
        "version": 1,
        "trafficType": "AROUND",
        "centerLat": sample_lat,
        "centerLon": sample_lon,
        "radius": 1,
        "zoomLevel": 14,
        "reqCoordType": "WGS84GEO",
        "resCoordType": "WGS84GEO",
    }
    headers = {"Accept": "application/json", "appKey": app_key}

    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print("[traffic] get_traffic_for_segment error:", e)
        return None

    features = data.get("features", [])
    if not features:
        return None

    # -------------------------
    # 1) roadType 필터 (고속도로만)
    # -------------------------
    if want_road_type is not None:
        def _rt_ok(f):
            rt = f.get("properties", {}).get("roadType")
            # "001" 이나 1 둘 다 허용
            if rt is None:
                return False
            if isinstance(rt, str):
                return rt == str(want_road_type)
            try:
                return int(rt) == int(want_road_type)
            except Exception:
                return False

        ft = [f for f in features if _rt_ok(f)]
        if ft:   # 하나라도 있으면 이걸로 교체
            features = ft

    # -------------------------
    # 2) 방향 필터 (상행/하행 → direction 0/1)
    # -------------------------
    dir_code = None
    if want_direction == "상행":
        dir_code = 0
    elif want_direction == "하행":
        dir_code = 1

    if dir_code is not None:
        filtered = [
            f for f in features
            if f.get("properties", {}).get("direction") == dir_code
        ]
        # 원하는 방향이 하나도 없다면 fallback 전체 사용
        if filtered:
            features = filtered

    if not features:
        return None

    # -------------------------
    # 3) 샘플 좌표에서 가장 가까운 링크 선택
    # -------------------------
    def dist_feature(f):
        geom = f.get("geometry", {})
        coords = geom.get("coordinates")
        if not coords:
            return float("inf")
        lon, lat = coords[0]
        return math.sqrt((lat - sample_lat) ** 2 + (lon - sample_lon) ** 2)

    best_feature = min(features, key=dist_feature)
    props = best_feature.get("properties", {})

    traffic_info = {
        "roadName": props.get("roadName"),
        "speed": props.get("speed"),
        "congestion": props.get("congestion"),
        "linkId": props.get("linkId"),
        "tm_direction": props.get("direction"),
        "roadType": props.get("roadType"),  # 디버그용으로 같이 넣어둠
        "description": props.get("description"),
        "updateTime": props.get("updateTime"),
    }
    return traffic_info




# =========================
# 5. bins + 실시간 교통 → segment_report
# =========================
def build_segment_report(bins, app_key):
    results = []

    for (line, s_ic, e_ic, direction), stats in bins.items():
        sample_lat, sample_lon = stats["sample_coord"]

        # 🔹 여기서 방향 정보 넘겨줌 + 고속도로만 보도록 roadType 고정
        traffic_info = get_traffic_for_segment(
            sample_lat,
            sample_lon,
            app_key,
            want_direction=direction,   # "상행" / "하행" / 방향미상
            want_road_type="001",       # 고속도로
        )

        row = {
            "line": line,
            "from_ic": s_ic,
            "to_ic": e_ic,
            "direction": direction,
            "normal": stats["normal"],
            "defect": stats["defect"],
            "total": stats["total"],
            "sample_coord": (sample_lat, sample_lon),
            "traffic": traffic_info,
            "image_ids": stats.get("image_ids", []),
            "defect_image_ids": stats.get("defect_image_ids", []),
        }
        results.append(row)

    print("[traffic] build_segment_report: num segments =", len(results))
    return results



# =========================
# 6. 외부에서 한 번에 쓰는 함수
# =========================
def analyze_segments(points, ic_df, app_key,
                     radius=200, opt=0, vehicle_type=0, top_k=5):
    """
    points      : [(lat, lon, status), ...]
    ic_df       : IC/JC CSV를 읽어온 DataFrame
    app_key     : Tmap APP KEY
    """
    ic_df_norm = normalize_ic_df(ic_df)
    bins = bin_segments(points, ic_df_norm, app_key,
                        radius=radius, opt=opt, vehicle_type=vehicle_type, top_k=top_k)
    report = build_segment_report(bins, app_key)
    return report
# =========================
# 7. "지금 당장" 보수 우선순위 랭킹 (시간대 CSV 없이 버전)
# =========================
def rank_segments_for_now(rows,
                          w_defect: float = 0.7,
                          w_congestion: float = 0.3):
    scored = []

    for r in rows:
        total = r.get("total", 0) or 0
        defect = r.get("defect", 0) or 0
        defect_ratio = defect / total if total > 0 else 0.0

        traffic = r.get("traffic") or {}
        cong_raw = traffic.get("congestion") if traffic else None

        # Tmap congestion: 1(원활) ~ 4(정체) 라고 가정
        if cong_raw is None:
            congestion_score = 0.5   # 데이터 없으면 중립값
        else:
            try:
                c = float(cong_raw)
                # 1 → 1.0, 4 → 0.0 쪽으로 매핑
                congestion_score = max(0.0, min(1.0, (4.5 - c) / 3.5))
            except Exception:
                congestion_score = 0.5

        score_now = w_defect * defect + w_congestion * congestion_score

        new_r = dict(r)
        new_r["defect_ratio"] = defect_ratio
        new_r["congestion_score"] = congestion_score
        new_r["score_now"] = score_now
        scored.append(new_r)

    scored.sort(key=lambda x: x["score_now"], reverse=True)
    print("[traffic] rank_segments_for_now: top scores =",
          [round(r["score_now"], 3) for r in scored[:3]])
    return scored
