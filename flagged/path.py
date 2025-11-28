

import pandas as pd
import json
import re
from tqdm import tqdm


class HybridTagsGenerator:

    def __init__(self):
        # Search intent patterns (E5 likes real text)
        self.search_intents = [
            "buy {item}",
            "best {item}",
            "{item} reviews",
        ]

    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r"[^\w\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -------------------------------------------------------
    # 1. Hierarchical tag boosting
    # -------------------------------------------------------
    def make_hierarchy_tags(self, path):
        levels = [l.strip() for l in path.split("/") if l.strip()]
        tags = []

        # Strong full-path signal
        full = " ".join(self.clean(l) for l in levels)
        tags.extend([full] * 8)   # <-- Strong boost

        # Progressive hierarchy
        for i in range(1, len(levels) + 1):
            seg = " ".join(self.clean(l) for l in levels[:i])
            tags.append(seg)

        # Parent-child reinforcement
        if len(levels) >= 2:
            parent = self.clean(levels[-2])
            child = self.clean(levels[-1])

            tags.extend([
                f"{parent} {child}",
                f"{child} {parent}",
                f"{child} in {parent}",
                f"{child} category {parent}"
            ])

        return tags

    # -------------------------------------------------------
    # 2. Extract key terms and word combos
    # -------------------------------------------------------
    def extract_terms(self, path):
        levels = [l.strip() for l in path.split("/") if l.strip()]
        terms = []

        for level in levels:
            cleaned = self.clean(level)
            if cleaned not in terms:
                terms.append(cleaned)

            words = [w for w in cleaned.split() if len(w) > 3]
            terms.extend(words)

            # bigrams for leaf and parent
            if level in levels[-2:]:
                for i in range(len(words) - 1):
                    terms.append(f"{words[i]} {words[i+1]}")

        # Remove duplicates, keep order
        return list(dict.fromkeys(terms))

    # -------------------------------------------------------
    # 3. Build final tag list for ONE category
    # -------------------------------------------------------
    def build_tags(self, category_id, category_path):
        tags = []

        # Hierarchy tags
        tags.extend(self.make_hierarchy_tags(category_path))

        # Key terms
        terms = self.extract_terms(category_path)
        tags.extend(terms[:15])

        # Search intent (for leaf level)
        leaf = self.clean(category_path.split("/")[-1])
        for pattern in self.search_intents[:2]:
            tags.append(pattern.format(item=leaf))

        # Clean + dedupe + limit
        seen = set()
        final = []

        for t in tags:
            c = self.clean(t)
            if c and c not in seen and len(c.split()) <= 6:
                seen.add(c)
                final.append(c)

        return final[:50]

    # -------------------------------------------------------
    # 4. Generate tags.json for entire CSV
    # -------------------------------------------------------
    def generate_tags_json(self, csv_path, output="tags.json"):
        df = pd.read_csv(csv_path, dtype=str)

        if "Category_ID" not in df.columns or "Category_path" not in df.columns:
            raise ValueError("CSV must contain Category_ID, Category_path columns")

        df = df.dropna(subset=["Category_path"])

        tags_dict = {}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building tags"):
            cid = str(row["Category_ID"])
            cpath = str(row["Category_path"])
            tags_dict[cid] = self.build_tags(cid, cpath)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(tags_dict, f, indent=2)

        print(f"✅ DONE: {output} saved.")
        return tags_dict


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python build_tags_json.py <categories.csv>")
        sys.exit()

    csv_file = sys.argv[1]
    gen = HybridTagsGenerator()
    gen.generate_tags_json(csv_file, "tags.json")
