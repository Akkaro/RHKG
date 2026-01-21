import os
from datasets import Features, Sequence, Value

MASTER_TAG_MAP = {
    "pers": "pers",
    "work": "work",
    "loc": "loc",
    "object": "O",
    "date": "date",
    "scope": "O",
    "PER": "pers",
    "LOC": "loc",
    "ORG": "org",
    "HumanProd": "O",
    "BUILDING": "loc",
    "STREET": "loc",
    "ALIEN": "O",
    "OTHER": "O",
    "FICTION": "O"
}


def standardize_tag(tag):
    if tag == "O" or tag == "_" or tag == "":
        return "O"

    prefix = ""
    label = tag
    if "-" in tag:
        prefix, label = tag.split("-", 1)
        prefix = prefix + "-"

    # Map the base label
    standard_label = MASTER_TAG_MAP.get(label, "O")

    if standard_label == "O":
        return "O"
    return f"{prefix}{standard_label}"


HIPE_FEATURES = Features(
    {
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "ne_coarse_lit": Sequence(Value("string")),
        "ne_coarse_meto": Sequence(Value("string")),
        "ne_fine_lit": Sequence(Value("string")),
        "ne_fine_meto": Sequence(Value("string")),
        "ne_fine_comp": Sequence(Value("string")),
        "ne_nested": Sequence(Value("string")),
        "nel_lit": Sequence(Value("string")),
        "nel_meto": Sequence(Value("string")),
        "misc": Sequence(Value("string")),
    }
)


def hipe_generator(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find data file at: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        guid = 0
        tokens = []
        ne_coarse_lit, ne_coarse_meto = [], []
        ne_fine_lit, ne_fine_meto, ne_fine_comp = [], [], []
        ne_nested, nel_lit, nel_meto, misc = [], [], [], []

        for line in f:
            line = line.strip()

            # 1. Skip Metadata & Header
            if line.startswith("#") or "TOKEN" in line:
                continue

            # 2. Skip empty lines
            if line == "":
                continue

            # 3. Parse Columns by Tab
            splits = line.split("\t")

            # Pad if missing columns
            if len(splits) < 10:
                splits += ["_"] * (10 - len(splits))

            tokens.append(splits[0])  # 0: TOKEN
            ne_coarse_lit.append(standardize_tag(splits[1]))  # 1: NE-COARSE-LIT
            ne_coarse_meto.append(splits[2])  # 2: NE-COARSE-METO
            ne_fine_lit.append(splits[3])  # 3: NE-FINE-LIT
            ne_fine_meto.append(splits[4])  # 4: NE-FINE-METO
            ne_fine_comp.append(splits[5])  # 5: NE-FINE-COMP
            ne_nested.append(splits[6])  # 6: NE-NESTED
            nel_lit.append(splits[7])  # 7: NEL-LIT
            nel_meto.append(splits[8])  # 8: NEL-METO

            # Clean up MISC column (sometimes has extra spaces)
            misc_val = splits[9].strip()  # 9: MISC
            misc.append(misc_val)

            # 4. Check for EndOfSentence Flag
            if "EndOfSentence" in misc_val:
                yield {
                    "id": str(guid),
                    "tokens": tokens,
                    "ne_coarse_lit": ne_coarse_lit,
                    "ne_coarse_meto": ne_coarse_meto,
                    "ne_fine_lit": ne_fine_lit,
                    "ne_fine_meto": ne_fine_meto,
                    "ne_fine_comp": ne_fine_comp,
                    "ne_nested": ne_nested,
                    "nel_lit": nel_lit,
                    "nel_meto": nel_meto,
                    "misc": misc,
                }
                guid += 1
                tokens = []
                ne_coarse_lit, ne_coarse_meto = [], []
                ne_fine_lit, ne_fine_meto, ne_fine_comp = [], [], []
                ne_nested, nel_lit, nel_meto, misc = [], [], [], []

        # Yield remaining tokens if file ends without flag
        if tokens:
            yield {
                "id": str(guid),
                "tokens": tokens,
                "ne_coarse_lit": ne_coarse_lit,
                "ne_coarse_meto": ne_coarse_meto,
                "ne_fine_lit": ne_fine_lit,
                "ne_fine_meto": ne_fine_meto,
                "ne_fine_comp": ne_fine_comp,
                "ne_nested": ne_nested,
                "nel_lit": nel_lit,
                "nel_meto": nel_meto,
                "misc": misc,
            }
