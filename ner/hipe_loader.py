import os
from datasets import Features, Sequence, Value

# 1. Updated Schema
# I added 'ne_fine_comp' to match your file structure.
HIPE_FEATURES = Features({
    "id": Value("string"),                          # Generated internally (not in file)
    "tokens": Sequence(Value("string")),
    "ne_coarse_lit": Sequence(Value("string")),
    "ne_coarse_meto": Sequence(Value("string")),
    "ne_fine_lit": Sequence(Value("string")),
    "ne_fine_meto": Sequence(Value("string")),
    "ne_fine_comp": Sequence(Value("string")),      # <--- ADDED THIS (Index 5)
    "ne_nested": Sequence(Value("string")),
    "nel_lit": Sequence(Value("string")),
    "nel_meto": Sequence(Value("string")),
    "misc": Sequence(Value("string"))
})

def hipe_generator(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find data file at: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        guid = 0
        # Initialize all buffers
        tokens = []
        ne_coarse_lit, ne_coarse_meto = [], []
        ne_fine_lit, ne_fine_meto, ne_fine_comp = [], [], []
        ne_nested, nel_lit, nel_meto, misc = [], [], [], []
        
        for line in f:
            line = line.strip()

            # 1. Skip Metadata & Header
            # We ignore your header row ("TOKEN ...") and metadata ("# ...")
            if line.startswith("#") or "TOKEN" in line:
                continue
            
            # 2. Skip empty lines (We rely on MISC flag now)
            if line == "":
                continue

            # 3. Parse Columns (Strict Mapping based on your snippet)
            splits = line.split("\t")
            
            # Pad if missing columns (standard safety)
            if len(splits) < 10:
                splits += ["_"] * (10 - len(splits))

            # MAPPING:
            # 0: TOKEN
            # 1: NE-COARSE-LIT
            # 2: NE-COARSE-METO
            # 3: NE-FINE-LIT
            # 4: NE-FINE-METO
            # 5: NE-FINE-COMP   <-- The column I missed before
            # 6: NE-NESTED
            # 7: NEL-LIT
            # 8: NEL-METO
            # 9: MISC
            
            tokens.append(splits[0])
            ne_coarse_lit.append(splits[1])
            ne_coarse_meto.append(splits[2])
            ne_fine_lit.append(splits[3])
            ne_fine_meto.append(splits[4])
            ne_fine_comp.append(splits[5]) 
            ne_nested.append(splits[6])
            nel_lit.append(splits[7])
            nel_meto.append(splits[8])
            
            # Clean up MISC column (sometimes has extra spaces)
            misc_val = splits[9].strip()
            misc.append(misc_val)

            # 4. Check for EndOfSentence Flag
            # As per your example: "InPrimaryReference|NoSpaceAfter|LED0.00" -> No Split
            # "EndOfSentence" -> Split
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
                    "misc": misc
                }
                guid += 1
                # Reset all buffers
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
                "misc": misc
            }