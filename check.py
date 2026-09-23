with open('arabic_freq_raw.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Take top 20k, extract word only
words = []
for line in lines[:20000]:
    parts = line.strip().split()
    if parts and all('\u0600' <= c <= '\u06FF' for c in parts[0]):
        words.append(parts[0])

with open('arabic_20k_words.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(words))

print(f"Saved {len(words)} frequency-ranked words")