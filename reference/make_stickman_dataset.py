# 추가 구현용 데이터 생성: COCO 관절 좌표 -> 졸라맨 이미지 + OpenPose 18관절 정답
# COCO annotations_trainval2017.zip 의 person_keypoints_train2017.json 필요
import json, os, random
import cv2
import numpy as np

ANN = "annotations/person_keypoints_train2017.json"
OUT = "stickman_data"
SIZE = 256
N_MAX = 5000

# COCO 17관절 -> OpenPose 18관절 순서 (None = 목, 양 어깨 중점으로 계산)
COCO2OP = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
BODY = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def wobbly_line(img, p1, p2, t):
    # 중간점을 살짝 흔들어서 손그림 느낌
    mid = (p1 + p2) / 2 + np.random.normal(0, 2, 2)
    pts = np.array([p1, mid, p2], np.int32)
    cv2.polylines(img, [pts], False, 0, t, cv2.LINE_AA)


def normalize(k):
    body = k[BODY]
    mn, mx = body.min(0), body.max(0)
    scale = SIZE * 0.75 / max((mx - mn).max(), 1)
    return (k - (mn + mx) / 2) * scale + SIZE / 2


def draw_stickman(k):
    img = np.full((SIZE, SIZE), 255, np.uint8)
    t = random.randint(2, 5)
    neck = (k[5] + k[6]) / 2
    pelvis = (k[11] + k[12]) / 2

    # 머리: 코 위치 중심의 원
    d = k[0] - neck
    dist = np.linalg.norm(d) + 1e-6
    r = min(dist * random.uniform(0.6, 0.9), SIZE * 0.12)
    center = (int(k[0][0]), int(k[0][1]))
    cv2.circle(img, center, int(r), 0, t, cv2.LINE_AA)
    wobbly_line(img, neck, k[0] - d / dist * r, t)

    # 몸통
    wobbly_line(img, neck, pelvis, t)

    # 팔 (30% 확률로 팔꿈치 생략)
    for e, w in [(7, 9), (8, 10)]:
        if random.random() < 0.3:
            wobbly_line(img, neck, k[w], t)
        else:
            wobbly_line(img, neck, k[e], t)
            wobbly_line(img, k[e], k[w], t)

    # 다리
    for kn, an in [(13, 15), (14, 16)]:
        wobbly_line(img, pelvis, k[kn], t)
        wobbly_line(img, k[kn], k[an], t)
    return img


def to_openpose(k, v):
    out = []
    for i in COCO2OP:
        if i is None:
            p = (k[5] + k[6]) / 2
        elif v[i] > 0:
            p = k[i]
        else:
            out.append([-1.0, -1.0])
            continue
        out.append([float(p[0]) / SIZE, float(p[1]) / SIZE])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(ANN) as f:
        anns = json.load(f)["annotations"]

    labels, count = {}, 0
    for a in anns:
        if a.get("iscrowd", 0) or a["num_keypoints"] < 13:
            continue
        kp = np.array(a["keypoints"], np.float32).reshape(17, 3)
        k, v = kp[:, :2], kp[:, 2]
        if (v[BODY] == 0).any():  # 몸 관절이 다 있는 사람만 사용
            continue
        k = normalize(k)
        name = f"{count:05d}.png"
        cv2.imwrite(os.path.join(OUT, name), draw_stickman(k))
        labels[name] = to_openpose(k, v)
        count += 1
        if count >= N_MAX:
            break

    with open(os.path.join(OUT, "labels.json"), "w") as f:
        json.dump(labels, f)
    print(count, "장 생성 완료")


if __name__ == "__main__":
    main()