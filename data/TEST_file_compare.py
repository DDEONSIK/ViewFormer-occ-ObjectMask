import zipfile
import os
import hashlib
from tqdm import tqdm  # 로딩 바를 위한 라이브러리

def calculate_file_hash(file_path, hash_type="sha256"):
    """
    파일의 해시 값을 계산하는 함수.
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

def verify_zip_and_extracted(zip_file_path, extracted_path):
    """
    압축 파일과 해제된 디렉토리의 파일 목록을 정밀 비교.
    경로 처리 문제를 해결하여 정확한 비교를 수행.

    Args:
    - zip_file_path (str): 압축 파일 경로
    - extracted_path (str): 압축 해제 디렉토리 경로
    """
    print(f"압축 파일 ({zip_file_path})과 해제된 파일 ({extracted_path})을 정밀 검사하는 중입니다...\n")
    
    # 압축 파일 목록 읽기
    print("압축 파일 목록을 읽는 중...")
    zip_files = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file in tqdm(zip_ref.infolist(), desc="압축 파일 정보 읽기 진행", unit="file"):
            if not file.is_dir():
                # root_folder 제거 및 경로 정규화
                relative_path = os.path.relpath(file.filename, zip_ref.namelist()[0].split('/')[0])
                zip_files[relative_path] = {
                    "size": file.file_size,
                    "crc": file.CRC
                }

    # 압축 해제된 파일 목록 읽기
    print("압축 해제된 파일 목록을 읽는 중...")
    extracted_files = {}
    for root, dirs, files in os.walk(extracted_path):
        for file in tqdm(files, desc="압축 해제 파일 정보 읽기 진행", unit="file"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, extracted_path)
            extracted_files[relative_path] = {
                "size": os.path.getsize(file_path),
                "hash": calculate_file_hash(file_path)
            }

    # 비교
    print("\n정밀 비교를 수행하는 중입니다...")
    missing_files = set(zip_files.keys()) - set(extracted_files.keys())
    extra_files = set(extracted_files.keys()) - set(zip_files.keys())

    mismatched_files = []
    for file in zip_files.keys() & extracted_files.keys():
        if zip_files[file]["size"] != extracted_files[file]["size"]:
            mismatched_files.append((file, "크기 불일치"))
        elif f"{zip_files[file]['crc']:08x}" != extracted_files[file]["hash"]:
            mismatched_files.append((file, "해시 불일치"))

    # 결과 출력
    print("\n=== 검사 결과 ===")
    print(f"1. 압축 파일 내 총 파일 수: {len(zip_files)}")
    print(f"2. 압축 해제된 총 파일 수: {len(extracted_files)}")
    print(f"3. 누락된 파일 수: {len(missing_files)}")
    print(f"4. 추가된 파일 수: {len(extra_files)}")
    print(f"5. 불일치 파일 수: {len(mismatched_files)}")

    if len(missing_files) == 0 and len(extra_files) == 0 and len(mismatched_files) == 0:
        print("\n결론: 압축 해제 및 검증이 정상적으로 완료되었습니다.")
    else:
        print("\n결론: 압축 해제에 문제가 발생했습니다.")

# 비교할 파일 및 디렉토리 설정
zip_file_path = "/home/hyun/local_storage/code/vieeew/ViewFormer-Occ/data/nuscenes/occ_flow_sparse_ext.zip"
extracted_path = "/home/hyun/local_storage/code/vieeew/ViewFormer-Occ/data/nuscenes/occ_flow_sparse_ext"

# 함수 호출
verify_zip_and_extracted(zip_file_path, extracted_path)
