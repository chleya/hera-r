#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick script to fix encoding issues in test files.
"""

import os
import re
from pathlib import Path

def fix_file_encoding(file_path):
    """Fix encoding issues in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace Unicode checkmarks with ASCII equivalents
    replacements = {
        '✅': '[OK]',
        '❌': '[FAIL]',
        '⚠️': '[WARN]',
        '🎉': '[SUCCESS]',
        '🔍': '[TEST]',
        '🔧': '[FIX]',
        '📊': '[STATS]',
        '🎯': '[GOAL]',
        '🛑': '[STOP]',
        '⚡': '[FAST]',
        '🧪': '[EXPERIMENT]',
        '📝': '[NOTE]',
        '🚀': '[LAUNCH]',
        '💡': '[IDEA]',
        '🔬': '[RESEARCH]',
        '🏗️': '[BUILD]',
        '🔍': '[INSPECT]',
        '📈': '[CHART]',
        '📉': '[DOWN]',
        '⚙️': '[GEAR]',
        '🔒': '[LOCK]',
        '🔓': '[UNLOCK]',
        '⏱️': '[TIMER]',
        '🎭': '[MASK]',
        '💥': '[BOOM]',
        '🌟': '[STAR]',
        '🔥': '[FIRE]',
        '💧': '[WATER]',
        '🌱': '[PLANT]',
        '🔄': '[REFRESH]',
        '📋': '[CLIPBOARD]',
        '🔗': '[LINK]',
        '📎': '[PAPERCLIP]',
        '📍': '[PIN]',
        '📌': '[PUSHPIN]',
        '🔖': '[BOOKMARK]',
        '🏷️': '[LABEL]',
        '💰': '[MONEY]',
        '💎': '[GEM]',
        '⚖️': '[SCALE]',
        '🔨': '[HAMMER]',
        '🛠️': '[TOOLS]',
        '🔧': '[WRENCH]',
        '🔩': '[NUT]',
        '⚒️': '[HAMMER_PICK]',
        '🪚': '[SAW]',
        '🔪': '[KNIFE]',
        '🏹': '[BOW]',
        '🛡️': '[SHIELD]',
        '🔫': '[GUN]',
        '💣': '[BOMB]',
        '🧨': '[FIRECRACKER]',
        '🔮': '[CRYSTAL]',
        '🧿': '[NAZAR]',
        '🎨': '[ART]',
        '🧵': '[THREAD]',
        '🧶': '[YARN]',
        '👓': '[GLASSES]',
        '🕶️': '[SUNGLASSES]',
        '🥽': '[GOGGLES]',
        '🥼': '[LABCOAT]',
        '🦺': '[SAFETYVEST]',
        '👔': '[NECKTIE]',
        '👕': '[TSHIRT]',
        '👖': '[JEANS]',
        '🧣': '[SCARF]',
        '🧤': '[GLOVES]',
        '🧥': '[COAT]',
        '🧦': '[SOCKS]',
        '👗': '[DRESS]',
        '👘': '[KIMONO]',
        '🥻': '[SARI]',
        '🩱': '[ONEPIECE]',
        '🩲': '[BRIEFS]',
        '🩳': '[SHORTS]',
        '👙': '[BIKINI]',
        '👚': '[BLOUSE]',
        '👛': '[PURSE]',
        '👜': '[HANDBAG]',
        '👝': '[CLUTCH]',
        '🎒': '[BACKPACK]',
        '👞': '[MANS_SHOE]',
        '👟': '[RUNNING_SHOE]',
        '🥾': '[HIKING_BOOT]',
        '🥿': '[FLAT_SHOE]',
        '👠': '[HIGH_HEEL]',
        '👡': '[SANDAL]',
        '🩴': '[THONG_SANDAL]',
        '👢': '[BOOT]',
        '👑': '[CROWN]',
        '👒': '[WOMANS_HAT]',
        '🎩': '[TOP_HAT]',
        '🎓': '[GRADUATION_CAP]',
        '🧢': '[BILLED_CAP]',
        '🪖': '[MILITARY_HELMET]',
        '⛑️': '[RESCUE_HELMET]',
        '📿': '[PRAYER_BEADS]',
        '💄': '[LIPSTICK]',
        '💍': '[RING]',
        '💎': '[GEM_STONE]',
    }
    
    for unicode_char, ascii_replacement in replacements.items():
        content = content.replace(unicode_char, ascii_replacement)
    
    # Add encoding fix to the beginning if not present
    if not content.startswith('# -*- coding: utf-8 -*-'):
        # Find the shebang line
        lines = content.split('\n')
        if lines[0].startswith('#!/usr/bin/env'):
            lines.insert(1, '# -*- coding: utf-8 -*-')
        else:
            lines.insert(0, '# -*- coding: utf-8 -*-')
        content = '\n'.join(lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")

def main():
    """Fix all test files in the project."""
    project_root = Path(__file__).parent
    
    # Files to fix
    test_files = [
        "test_core_functionality.py",
        "test_enhanced_features.py",
        "test_simple.py",
        "test_structure.py",
        "test_structure_simple.py",
    ]
    
    # Also check for any other Python files with tests
    for py_file in project_root.glob("**/*.py"):
        if "test" in py_file.name.lower():
            test_files.append(str(py_file.relative_to(project_root)))
    
    # Fix each file
    for test_file in test_files:
        file_path = project_root / test_file
        if file_path.exists():
            try:
                fix_file_encoding(file_path)
            except Exception as e:
                print(f"Error fixing {test_file}: {e}")
    
    print("\n" + "=" * 60)
    print("Encoding fixes complete!")
    print("=" * 60)
    print("\nAll test files have been updated to use ASCII characters")
    print("and include proper encoding declarations.")

if __name__ == "__main__":
    main()