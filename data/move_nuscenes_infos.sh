#!/bin/bash

# 원본 파일 경로
SRC_BASE="/home/hyun/local_storage/code/vieeew/ViewFormer-Occ/data/nuscenes_old"

# 대상 디렉토리 경로
DEST_BASE="/mnt/dataset_storage/nuscenes"

# 이동할 파일 목록
FILES=("nuscenes_infos_temporal_test.pkl"
       "nuscenes_infos_temporal_train.pkl"
       "nuscenes_infos_temporal_val.pkl")

echo "파일 이동을 시작합니다..."

# 이동 및 검증
for FILE in "${FILES[@]}"; do
    SRC="$SRC_BASE/$FILE"
    DEST="$DEST_BASE/$FILE"

    echo "파일 이동 중: $SRC -> $DEST"

    # 디렉토리 생성
    mkdir -p "$(dirname "$DEST")"

    # 파일 이동
    mv "$SRC" "$DEST" || { echo "파일 이동 실패: $SRC"; exit 1; }

    # 무결성 확인
    echo "무결성 확인 중: $DEST"
    if [ ! -f "$DEST" ]; then
        echo "무결성 확인 실패: $DEST가 존재하지 않습니다."
        exit 1
    fi
done

echo "모든 파일 이동이 성공적으로 완료되었습니다!"

# chmod +x move_nuscenes_infos.sh
# ./move_nuscenes_infos.sh
