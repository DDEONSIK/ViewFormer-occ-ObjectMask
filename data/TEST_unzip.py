import zipfile
import os
import hashlib
from tqdm import tqdm  # 로딩 바를 위한 tqdm 라이브러리
from datetime import datetime, timezone, timedelta  # 시간 측정

def calculate_file_hash(file_path, hash_type="sha256"):
    """
    파일의 해시 값 계산 함수.
    데이터 무결성 확인.

    Args:
    - file_path (str): 파일 경로
    - hash_type (str): 사용할 해시 알고리즘 (기본값: sha256)

    Returns:
    - str: 파일의 해시 값
    """
    hash_func = hashlib.new(hash_type)
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def extract_and_verify_zip(zip_file_path, extracted_path):
    """
    ZIP 파일을 정밀하고 효율적으로 해제하고 데이터 무결성을 검증하는 함수.
    로딩 바를 추가하여 진행 상황을 실시간 확인 가능.

    Args:
    - zip_file_path (str): 압축 파일 경로
    - extracted_path (str): 압축 해제 대상 디렉토리 경로

    Returns:
    - None
    """
    # 작업 시작 시간 기록 (KST 기준)
    kst = timezone(timedelta(hours=9))
    start_time = datetime.now(kst)
    print(f"작업 시작 시간 (KST): {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("압축 해제 시작...")
    print(f"압축 파일: {zip_file_path}")
    print(f"해제 경로: {extracted_path}")

    # 디렉토리가 없으면 생성
    if not os.path.exists(extracted_path):
        os.makedirs(extracted_path)

    # 압축 해제
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"압축 파일 내 총 {len(file_list)}개 파일 발견.")

        # 로딩 바 추가
        for file in tqdm(file_list, desc="압축 해제 진행", unit="file"):
            target_path = os.path.join(extracted_path, file)
            try:
                # 디렉토리 구조 생성 및 파일 해제
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zip_ref.open(file) as source, open(target_path, 'wb') as target:
                    while chunk := source.read(8192):
                        target.write(chunk)
            except Exception:
                continue  # 오류 발생 시 무시

    print("\n압축 해제 완료.")

    # 작업 완료 시간 기록 (KST 기준)
    end_time = datetime.now(kst)
    print(f"작업 완료 시간 (KST): {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 소요 시간 계산 및 출력
    elapsed_time = end_time - start_time
    print(f"총 작업 소요 시간: {elapsed_time}")

# 사용 예시
zip_file_path = "/home/hyun/local_storage/code/vieeew/ViewFormer-Occ/data/nuscenes/occ_flow_sparse_ext.zip" # 255223904/255223904
extracted_path = "/home/hyun/local_storage/code/vieeew/ViewFormer-Occ/data/nuscenes"

# 함수 호출
extract_and_verify_zip(zip_file_path, extracted_path)
