#!/usr/bin/env python3
"""
GQA 데이터셋 → vllm bench serve 용 sharegpt JSON 변환 (로컬 이미지 경로)

Usage:
  python convert_gqa.py --max-images 398 --output-dir bench-dataset
"""
import argparse
import json
import random
import os
from collections import defaultdict
from datasets import load_dataset

# vLLM docker 내부 경로 기준
CONTAINER_IMAGE_DIR = "/bench-dataset/gqa_data/gqa_images"


def build_entry(img_id: str, question: str, answer: str, idx: int) -> dict:
    # sharegpt4v_coco 포맷과 동일하게
    return {
        "id": f"{img_id}_{idx}",
        "image": f"{CONTAINER_IMAGE_DIR}/{img_id}.jpg",
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt",   "value": answer},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=398)
    parser.add_argument("--output-dir", type=str, default="./gqa_data")
    args = parser.parse_args()

    # ── 데이터 로드 ──
    print("Loading images...")
    images_ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    print("Loading instructions...")
    instructions_ds = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")

    # ── 이미지 파일로 저장 ──
    img_dir = f"{args.output_dir}/gqa_images"
    os.makedirs(img_dir, exist_ok=True)

    image_ids = [item["id"] for item in images_ds]
    selected_ids = set(image_ids[: args.max_images])

    print(f"Saving {len(selected_ids)} images to {img_dir}/...")
    saved = 0
    for item in images_ds:
        if item["id"] in selected_ids:
            item["image"].save(f"{img_dir}/{item['id']}.jpg")
            saved += 1
    print(f"Saved {saved} images")

    # ── 질문을 이미지별 그루핑 ──
    grouped = defaultdict(list)
    for item in instructions_ds:
        img_id = item["imageId"]
        if img_id in selected_ids:
            # fullAnswer 사용 ("no" 대신 "No, it is clear." 같은 문장)
            # → vLLM sharegpt 파서가 짧은 응답을 필터링하는 문제 방지
            answer = item["fullAnswer"]
            if not answer or len(answer.split()) < 3:
                answer = f"The answer is: {item['answer']}."
            grouped[img_id].append({
                "question": item["question"],
                "answer": answer,
            })

    # ── sorted (이미지별 연속 배치 → prefix cache hit 극대화) ──
    sorted_entries = []
    for img_id, qas in grouped.items():
        for idx, qa in enumerate(qas):
            sorted_entries.append(build_entry(img_id, qa["question"], qa["answer"], idx))

    sorted_path = f"{args.output_dir}/gqa_sorted.json"
    with open(sorted_path, "w") as f:
        json.dump(sorted_entries, f, ensure_ascii=False)

    # ── shuffled (baseline) ──
    shuffled_entries = sorted_entries.copy()
    random.shuffle(shuffled_entries)

    shuffled_path = f"{args.output_dir}/gqa_shuffled.json"
    with open(shuffled_path, "w") as f:
        json.dump(shuffled_entries, f, ensure_ascii=False)

    # ── 통계 + 샘플 확인 ──
    n_images = len(grouped)
    n_questions = len(sorted_entries)
    sorted_kb = os.path.getsize(sorted_path) / 1024

    print(f"\n📋 샘플 확인:")
    for e in sorted_entries[:3]:
        q = e["conversations"][0]["value"]
        a = e["conversations"][1]["value"]
        print(f"   Q: {q[:60]}")
        print(f"   A: {a}")
        print()

    print(f"✅ 완료!")
    print(f"   이미지:  {n_images}개 (→ {img_dir}/)")
    print(f"   질문:    {n_questions}개 (avg {n_questions/n_images:.1f}/image)")
    print(f"   JSON:    {sorted_kb:.0f} KB")
    print(f"   → {sorted_path}")
    print(f"   → {shuffled_path}")


if __name__ == "__main__":
    main()