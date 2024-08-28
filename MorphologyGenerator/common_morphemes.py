def extract_affixes_from_lexc(lexc_file_path):
    prefixes = set()
    suffixes = set()
    
    with open(lexc_file_path, 'r') as lexc_file:
        for line in lexc_file:
            line = line.strip()
            
            # Skip empty lines or comments
            if not line or line.startswith('!'):
                continue

            # Extract prefixes
            if line.startswith('+'):
                parts = line.split()
                for part in parts:
                    if ':' in part or '^' in part:
                        prefix = part.split(':')[0].split('^')[0]
                        if prefix.startswith('+'):
                            prefixes.add(prefix)

            # Extract suffixes
            if '^' in line or ':' in line:
                suffix_candidates = line.split()
                for suffix_candidate in suffix_candidates:
                    if ':' in suffix_candidate:
                        suffix = suffix_candidate.split(':')[1]
                        if suffix != '0':
                            suffixes.add(suffix)
                    if '^' in suffix_candidate:
                        suffix = suffix_candidate.split('^')[1]
                        suffixes.add(suffix)
    
    return prefixes, suffixes

# Example usage
lexc_file_path = 'path/to/your/kin.lexc'
prefixes, suffixes = extract_affixes_from_lexc(lexc_file_path)

print("Extracted Prefixes:", prefixes)
print("Extracted Suffixes:", suffixes)
