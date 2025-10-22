# convert_csv_to_json.py
import csv
import json

def convert_csv_to_json(csv_file, json_file):
    print(f'Reading CSV: {csv_file}')
    
    categories = []
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        
        # Skip header row
        header = next(reader)
        print(f'Header columns: {len(header)}')
        
        row_count = 0
        for row in reader:
            if not row or len(row) < 3:
                continue
            
            category_id = str(row[0]).strip()
            category_path = str(row[1]).strip()
            
            if not category_id or not category_path:
                continue
            
            # Build category object
            category = {
                'Category_ID': category_id,
                'Category_path': category_path
            }
            
            # ONLY extract columns 2-11 (Level 1-10)
            levels = []
            max_level_column = min(12, len(row))
            
            for i in range(2, max_level_column):
                level_value = str(row[i]).strip()
                
                if not level_value or level_value.lower() in ['', 'nan', 'null', 'none']:
                    continue
                
                level_num = i - 1
                category[f'level_{level_num}'] = level_value
                levels.append(level_value)
                category[f'level_{level_num}_path'] = '/'.join(levels)
            
            category['depth'] = len(levels)
            categories.append(category)
            row_count += 1
            
            if row_count % 5000 == 0:
                print(f'  Processed {row_count} categories...')
    
    print(f'\nTotal categories: {len(categories)}')
    
    # Depth distribution
    from collections import Counter
    depths = Counter(cat['depth'] for cat in categories)
    print('\nDepth distribution:')
    for depth in sorted(depths.keys()):
        print(f'  Level {depth}: {depths[depth]} categories')
    
    # Save
    print(f'\nWriting: {json_file}')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    
    print(f'\nDone! Total: {len(categories)} categories')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python convert_csv_to_json.py <input.csv> [output.json]')
        sys.exit(1)
    
    csv_file = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else 'data/categories.json'
    convert_csv_to_json(csv_file, json_file)
