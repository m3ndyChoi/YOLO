"""
YOLOv5 format (XYWH) to XYXY format converter
Converts: class x_center y_center width height -> class x_min y_min x_max y_max
"""

import os
from pathlib import Path
from tqdm import tqdm
import shutil

def convert_yolo_v5_to_xyxy(label_path, backup=True):
    """
    Converts a single label file from YOLOv5 format (class xc yc w h)
    to XYXY format (class x1 y1 x2 y2).
    
    Args:
        label_path: Path to the label file
        backup: If True, creates a backup before conversion
    
    Returns:
        tuple: (success, message)
    """
    try:
        # Backup original file
        if backup:
            backup_path = Path(str(label_path) + '.backup')
            if not backup_path.exists():
                shutil.copy2(label_path, backup_path)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        converted = False
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                new_lines.append('\n')
                continue
                
            parts = line.split()
            
            # Check if already in XYXY format
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    
                    # Check if it's XYWH (center + size) or XYXY (corners)
                    # XYXY: x2 > x1 and y2 > y1
                    # XYWH: width and height are typically < 1
                    if coords[2] > coords[0] and coords[3] > coords[1]:
                        # Already XYXY format
                        new_lines.append(line + '\n')
                        continue
                    
                    # Convert XYWH to XYXY
                    xc, yc, w, h = coords
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2
                    
                    # Clamp values to [0, 1]
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))
                    
                    # Validate box
                    if x2 <= x1 or y2 <= y1:
                        print(f"  ⚠️  Warning: Invalid box at line {line_num}: {line}")
                        continue
                    
                    new_lines.append(f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")
                    converted = True
                    
                except ValueError as e:
                    print(f"  ❌ Error parsing line {line_num}: {line}")
                    new_lines.append(line + '\n')
                    
            else:
                # Keep non-standard lines as is
                new_lines.append(line + '\n')
        
        # Write converted data
        with open(label_path, 'w') as f:
            f.writelines(new_lines)
        
        return True, "Converted" if converted else "Already XYXY"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def validate_xyxy_format(label_path):
    """
    Validates that label file is in correct XYXY format.
    
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
                continue
            
            try:
                cls = int(parts[0])
                x1, y1, x2, y2 = [float(x) for x in parts[1:5]]
                
                # Validate ranges
                if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                    errors.append(f"Line {line_num}: Coordinates out of range [0,1]")
                
                # Validate order
                if x2 <= x1 or y2 <= y1:
                    errors.append(f"Line {line_num}: Invalid box (x2<=x1 or y2<=y1)")
                
            except ValueError:
                errors.append(f"Line {line_num}: Invalid number format")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        return False, [f"File error: {str(e)}"]

def main():
    print("="*60)
    print("YOLOv5 (XYWH) to XYXY Format Converter")
    print("="*60)
    
    # Get dataset root path
    dataset_root = input("\n📁 Enter dataset root path: ").strip().strip('"')
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        print(f"❌ Error: Path '{dataset_path}' does not exist.")
        return
    
    # Ask for backup option
    backup = input("\n💾 Create backup files? (y/n, default=y): ").strip().lower() != 'n'
    
    # Ask for validation
    validate = input("✅ Validate after conversion? (y/n, default=y): ").strip().lower() != 'n'
    
    # Find all .txt files
    label_files = list(dataset_path.rglob("*.txt"))
    
    # Exclude common metadata files
    exclude_names = ['classes.txt', 'README.txt', 'train.txt', 'val.txt', 'test.txt']
    label_files = [f for f in label_files if f.name not in exclude_names]
    
    print(f"\n📊 Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        print("❌ No label files found!")
        return
    
    # Ask for confirmation
    confirm = input(f"\n⚠️  Convert {len(label_files)} files? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Conversion cancelled.")
        return
    
    print("\n🔄 Converting labels...\n")
    
    # Statistics
    stats = {
        'converted': 0,
        'already_xyxy': 0,
        'failed': 0,
        'validation_errors': 0
    }
    
    # Convert files
    for label_file in tqdm(label_files, desc="Converting"):
        success, message = convert_yolo_v5_to_xyxy(label_file, backup=backup)
        
        if not success:
            stats['failed'] += 1
            print(f"\n❌ {label_file.name}: {message}")
        elif "Converted" in message:
            stats['converted'] += 1
        else:
            stats['already_xyxy'] += 1
    
    # Validation
    if validate:
        print("\n\n✅ Validating converted files...\n")
        for label_file in tqdm(label_files, desc="Validating"):
            is_valid, errors = validate_xyxy_format(label_file)
            if not is_valid:
                stats['validation_errors'] += 1
                print(f"\n⚠️  {label_file.name}:")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"    {error}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Conversion Summary")
    print("="*60)
    print(f"✅ Successfully converted: {stats['converted']}")
    print(f"ℹ️  Already XYXY format:   {stats['already_xyxy']}")
    print(f"❌ Failed:                 {stats['failed']}")
    if validate:
        print(f"⚠️  Validation errors:     {stats['validation_errors']}")
    print("="*60)
    
    if backup:
        print(f"\n💾 Backup files saved with .backup extension")
    
    print(f"\n✨ Done! Total processed: {len(label_files)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Conversion cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
