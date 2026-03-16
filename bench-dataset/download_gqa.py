# download_gqa.py
from datasets import load_dataset
import json
import os
from collections import defaultdict

# 1) 이미지와 질문 다운로드
print("Downloading images...")
images_ds = load_dataset(
    "lmms-lab/GQA", 
    "testdev_balanced_images",
    split="testdev"
)

print("Downloading instructions...")
instructions_ds = load_dataset(
    "lmms-lab/GQA",
    "testdev_balanced_instructions", 
    split="testdev"
)
# instructions_ds 로드 직후에
print(instructions_ds[0])
# 2) 이미지 저장
os.makedirs("gqa_data/images", exist_ok=True)
image_map = {}
for item in images_ds:
    img_id = item["id"]
    img_path = f"gqa_data/images/{img_id}.jpg"
    item["image"].save(img_path)
    image_map[img_id] = img_path

print(f"Saved {len(image_map)} images")

# 3) 질문을 이미지별로 그루핑 (prefix caching 최적화 핵심!)
grouped = defaultdict(list)
for item in instructions_ds:
    img_id = item["imageId"]  # 이미지 ID 참조
    grouped[img_id].append({
        "question_id": item["id"],
        "question": item["question"],
        "answer": item.get("answer", ""),
    })

# 4) 이미지별 정렬된 요청 리스트 생성
#    같은 이미지의 쿼리를 연속 배치 → prefix cache hit 극대화
requests = []
for img_id, questions in grouped.items():
    for q in questions:
        requests.append({
            "image_id": img_id,
            "image_path": f"gqa_data/images/{img_id}.jpg",
            "question_id": q["question_id"],
            "question": q["question"],
            "answer": q["answer"],
        })

with open("gqa_data/requests_sorted.json", "w") as f:
    json.dump(requests, f, indent=2)

# 셔플 버전도 만들어 (baseline 비교용)
import random
shuffled = requests.copy()
random.shuffle(shuffled)
with open("gqa_data/requests_shuffled.json", "w") as f:
    json.dump(shuffled, f, indent=2)

print(f"Total requests: {len(requests)}")
print(f"Unique images: {len(grouped)}")
print(f"Avg queries/image: {len(requests)/len(grouped):.1f}")