# import re
# from collections import defaultdict

# def parse_lexc_file(lexc_file_path):
#     morphemes_by_category = defaultdict(list)
#     current_category = None

#     with open(lexc_file_path, 'r') as lexc_file:
#         for line in lexc_file:
#             line = line.strip()
            
#             # Identify the start of a new category
#             if line.startswith('LEXICON'):
#                 parts = line.split()
#                 if len(parts) > 1:
#                     current_category = parts[1]
            
#             # Extract morphemes within the category
#             elif current_category and line:
#                 morpheme = re.split(r'\s+', line)[0]  # Take the first part of the line as morpheme
#                 morphemes_by_category[current_category].append(morpheme)
    
#     return morphemes_by_category

# # Example usage
# lexc_file_path = 'path/to/kin.lexc'
# morphemes_by_category = parse_lexc_file(lexc_file_path)
