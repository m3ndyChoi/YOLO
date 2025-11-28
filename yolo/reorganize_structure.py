# reorganize_to_yolo_structure.py
"""
COCO 스타일 구조를 YOLO 표준 구조로 변경

Before:
  dataset/
  ├─ Train/
  │   ├─ image1.jpg
  │   └─ label1.txt
  ├─ Valid/
  └─ Test/

After:
  dataset/
  ├─ images/
  │   ├─ Train/
  │   ├─ Valid/
  │   └─ Test/
  └─ labels/
      ├─ Train/
      ├─ Valid/
      └─ Test/
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def reorganize_dataset(dataset_root, backup=True):
    """
    데이터셋 구조를 YOLO 표준 형식으로 변경
    
    Args:
        dataset_root: 데이터셋 루트 경로
        backup: True면 원본 백업
    """
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        print(f"❌ Error: {dataset_path} 경로가 존재하지 않습니다.")
        return
    
    print("="*60)
    print("YOLO Dataset Structure Reorganizer")
    print("="*60)
    
    # 백업
    if backup:
        backup_path = Path(str(dataset_path) + "_backup")
        if not backup_path.exists():
            print(f"\n💾 Creating backup: {backup_path}")
            shutil.copytree(dataset_path, backup_path)
            print("✅ Backup complete")
        else:
            print(f"\n⚠️  Backup already exists: {backup_path}")
    
    # 새 폴더 생성
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"
    
    images_root.mkdir(exist_ok=True)
    labels_root.mkdir(exist_ok=True)
    
    # Train, Valid, Test 폴더 처리
    splits = ["Train", "Valid", "Test"]
    
    # 대소문자 변형도 체크
    found_splits = []
    for split in splits:
        split_path = dataset_path / split
        if split_path.exists() and split_path.is_dir():
            found_splits.append(split)
        else:
            # 소문자 버전 체크
            split_lower = dataset_path / split.lower()
            if split_lower.exists() and split_lower.is_dir():
                found_splits.append(split.lower())
    
    if not found_splits:
        print(f"\n❌ Error: Train, Valid, Test 폴더를 찾을 수 없습니다.")
        print(f"현재 경로: {dataset_path}")
        print(f"하위 폴더: {[f.name for f in dataset_path.iterdir() if f.is_dir()]}")
        return
    
    print(f"\n📁 Found splits: {found_splits}")
    
    # 통계
    stats = {
        'images_moved': 0,
        'labels_moved': 0,
        'errors': 0
    }
    
    # 각 split 처리
    for split in found_splits:
        print(f"\n🔄 Processing {split}...")
        
        old_split_path = dataset_path / split
        
        # 새 경로 생성
        new_images_path = images_root / split
        new_labels_path = labels_root / split
        
        new_images_path.mkdir(exist_ok=True)
        new_labels_path.mkdir(exist_ok=True)
        
        # 파일 확장자 정의
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        label_extensions = {'.txt'}
        
        # 파일 수집
        all_files = list(old_split_path.rglob("*"))
        files_to_process = [f for f in all_files if f.is_file()]
        
        print(f"  Found {len(files_to_process)} files")
        
        # 파일 이동
        for file_path in tqdm(files_to_process, desc=f"  Moving {split}"):
            try:
                # annotations.json 같은 메타파일 제외
                if file_path.name in ['annotations.json', 'classes.txt', 'README.txt']:
                    continue
                
                file_ext = file_path.suffix.lower()
                
                # 이미지 파일
                if file_ext in image_extensions:
                    dest_path = new_images_path / file_path.name
                    shutil.copy2(file_path, dest_path)
                    stats['images_moved'] += 1
                
                # 라벨 파일
                elif file_ext in label_extensions:
                    dest_path = new_labels_path / file_path.name
                    shutil.copy2(file_path, dest_path)
                    stats['labels_moved'] += 1
                
            except Exception as e:
                print(f"\n  ❌ Error processing {file_path.name}: {e}")
                stats['errors'] += 1
        
        print(f"  ✅ {split} complete")
    
    # 최종 통계
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    print(f"✅ Images moved:  {stats['images_moved']}")
    print(f"✅ Labels moved:  {stats['labels_moved']}")
    print(f"❌ Errors:        {stats['errors']}")
    print("="*60)
    
    print("\n📂 New structure:")
    print(f"  {dataset_path}/")
    print(f"  ├─ images/")
    for split in found_splits:
        img_count = len(list((images_root / split).glob("*")))
        print(f"  │   ├─ {split}/ ({img_count} files)")
    print(f"  └─ labels/")
    for split in found_splits:
        lbl_count = len(list((labels_root / split).glob("*.txt")))
        print(f"      ├─ {split}/ ({lbl_count} files)")
    
    # 원본 폴더 삭제 확인
    print(f"\n⚠️  Original folders (Train, Valid, Test) are still in place.")
    delete = input("Delete original folders? (y/n, default=n): ").strip().lower()
    
    if delete == 'y':
        for split in found_splits:
            old_path = dataset_path / split
            if old_path.exists():
                shutil.rmtree(old_path)
                print(f"  🗑️  Deleted {split}/")
        print("✅ Original folders deleted")
    else:
        print("ℹ️  Original folders kept (can delete manually later)")
    
    print(f"\n✨ Done! Dataset reorganized to YOLO structure.")
    if backup:
        print(f"💾 Backup saved at: {backup_path}")


def verify_structure(dataset_root):
    """
    변환된 구조 검증
    """
    dataset_path = Path(dataset_root)
    
    print("\n🔍 Verifying structure...")
    
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"
    
    if not images_root.exists() or not labels_root.exists():
        print("❌ images/ or labels/ folder not found")
        return False
    
    splits = ["Train", "Valid", "Test"]
    
    for split in splits:
        img_path = images_root / split
        lbl_path = labels_root / split
        
        # 소문자 버전도 체크
        if not img_path.exists():
            img_path = images_root / split.lower()
        if not lbl_path.exists():
            lbl_path = labels_root / split.lower()
        
        if img_path.exists() and lbl_path.exists():
            img_files = set([f.stem for f in img_path.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
            lbl_files = set([f.stem for f in lbl_path.glob("*.txt")])
            
            print(f"  {split}:")
            print(f"    Images: {len(img_files)}")
            print(f"    Labels: {len(lbl_files)}")
            
            # 매칭 확인
            matched = img_files & lbl_files
            unmatched_images = img_files - lbl_files
            unmatched_labels = lbl_files - img_files
            
            print(f"    Matched: {len(matched)}")
            if unmatched_images:
                print(f"    ⚠️  Images without labels: {len(unmatched_images)}")
            if unmatched_labels:
                print(f"    ⚠️  Labels without images: {len(unmatched_labels)}")
    
    print("✅ Verification complete")
    return True


if __name__ == "__main__":
    print("YOLO Dataset Structure Reorganizer")
    print("="*60)
    
    # 경로 입력
    default_path = "D:/PVS-2025/2. 프로젝트/2025/2509_화승알앤에이/마킹-학습데이터/Dataset-cocoformat"
    dataset_root = input(f"\n📁 Enter dataset root path\n   (default: {default_path})\n   > ").strip().strip('"') or default_path
    
    # 백업 옵션
    backup = input("\n💾 Create backup? (y/n, default=y): ").strip().lower() != 'n'
    
    # 실행
    reorganize_dataset(dataset_root, backup=backup)
    
    # 검증
    verify = input("\n🔍 Verify new structure? (y/n, default=y): ").strip().lower() != 'n'
    if verify:
        verify_structure(dataset_root)
    
    print("\n✨ All done!")
