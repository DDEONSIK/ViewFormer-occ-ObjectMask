#!/bin/bash

# 원본 디렉토리 경로
SRC_BASE="/home/hyun/local_storage/code/vieeew/ViewFormer-Occ/data/nuscenes_old"

# 대상 디렉토리 경로
DEST_BASE="/mnt/dataset_storage/nuscenes"

# 복사할 디렉토리 목록
DIRS=("v1.0-mini" "v1.0-test" "v1.0-trainval" "gts")

echo "복사를 시작합니다..."

# 복사 및 검증
for DIR in "${DIRS[@]}"; do
    SRC="$SRC_BASE/$DIR"
    DEST="$DEST_BASE/$DIR"

    echo "복사 중: $SRC -> $DEST"

    # 디렉토리 생성 및 복사
    mkdir -p "$DEST"
    rsync -r --progress "$SRC/" "$DEST/" || { echo "복사 실패: $SRC"; exit 1; }

    # 무결성 확인
    echo "무결성 확인 중: $SRC -> $DEST"
    diff -r "$SRC" "$DEST" > /dev/null || { echo "무결성 확인 실패: $SRC"; exit 1; }
done

echo "복사가 성공적으로 완료되었습니다!"


# chmod +x copy_and_verify.sh
# ./copy_and_verify.sh
