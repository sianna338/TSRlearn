import random
import pandas as pd
from collections import deque
from typing import List, Dict
from openpyxl.workbook import Workbook
import os
import math
from collections import defaultdict
import numpy as np

# --- settings ---

n_participants = 30
base_name = "localiser_conditions" # root filename without ppt index

SEED = 7 # seed for first participant, might be better to randomise 
N_EXEMPLARS_PER_CONCEPT = 40
MISMATCH_RATIO = 0.20 # ratio of word-image mismatch pairs 
concepts = [
    "bug","berry","bicycle","bird","box","car","chair","coffee","dog","face","fish",
    "guitar","hammer","hand","house","jacket","pencil","phone","pizza","plane","tree"
]
left_key = "g" # key to choose the option presented on the left 
right_key = "b"
screen_positions = ["left", "right"] # positions on the screen with response options 


folder_name_map = {c: c for c in concepts}
german_map = {
    "bug":"Käfer","berry":"Beere","bicycle":"Fahrrad","bird":"Vogel","box":"Kiste",
    "car":"Auto","chair":"Stuhl", "coffee":"Kaffee","dog":"Hund","face":"Gesicht","fish":"Fisch",
    "guitar":"Gitarre","hammer":"Hammer","hand":"Hand","house":"Haus","jacket":"Jacke",
    "pencil":"Bleistift","phone":"Handy","pizza":"Pizza","plane":"Flugzeug","tree":"Baum",
} # map english to german words

MAX_ATTEMPTS = 5000 # max. attempts to build trial list that matches constraints


# --- contraints for generating the condition file --- 

# constraint 01: not more than x identical concepts in a row (for words and images)
MAX_STREAK = 1 

# constraint 02: per-concept mismatches 
PER_CONCEPT_MISMATCH = int(MISMATCH_RATIO * N_EXEMPLARS_PER_CONCEPT)

# constraint 03: try to spread the exemplars of one concept evenly 
target_per_quartile = N_EXEMPLARS_PER_CONCEPT // 4  

# constraint 04: make sure not always same images get paired with same words on mismatch trials

# constraint 05: make sure mismatches are equally enough distributed
MISMATCH_PER_QUARTILE = int(round((PER_CONCEPT_MISMATCH*len(concepts)) // 4))
QUARTILE_TOL = 3  # allow deviation

# constraint 06 (in case key-resp mappings change during the experiment): 
# same button is not more than x times in a row the right answer 
MAX_STREAK_COR_KEY = 3

# --- additionally: pick ITI

# add uniform ITI within these limits 
ITI_RANGE = (0.75, 1.25) 

# --- function to find non-matching words from the list of all words --- 

# this is random choice now, should we make it more systematic?
def pick_nonmatching_word(concept: str, used_mismatches: dict, rng: random.Random) -> str:

    candidates = [w for w in concepts if w != concept]

    # constraint 04
    candidates = [w for w in candidates if w not in used_mismatches[concept]]

    # if we exhausted all options, reset memory for this concept
    if not candidates:
        used_mismatches[concept].clear()
        candidates = [w for w in concepts if w != concept]

    # pick one and remember it
    chosen = rng.choice(candidates)
    used_mismatches[concept].add(chosen)
    return chosen


# --- build valid trial list (take care of constraints) ---

def build_valid_schedule(trials: List[Dict],
                         rng: random.Random,
                         max_attempts: int,
                         max_streak: int,
                         n_exemplars_per_concept: int):
    """
    Enforces:
      - MAX_STREAK for concepts & words (constraint 01)
      - exemplar spreading: each concept ~ evenly per quartile (constraint 03)
    """
    total = len(trials)

    # constraint 03, spread concepts evenly across quartiles 
    # build dict with concept names and 4-slot counter of occurrence per quartile 
    counts_quartile = {c: [0, 0, 0, 0] for c in [t["concept"] for t in trials][::n_exemplars_per_concept]}
    
    # loop over max constraints
    for _ in range(max_attempts):
        remaining = trials[:]
        rng.shuffle(remaining)
        schedule = []

        # store items from previous trials to avoid repeats 
        last_concepts = deque(maxlen=max_streak)
        last_words    = deque(maxlen=max_streak)

        # reset quartile counters
        for c in counts_quartile:
            counts_quartile[c] = [0,0,0,0]

        # constraint 05, spread mismatches evenly across quartiles 
        # build 4-slot counter of occurrence per quartile 
        counts_mismatch_quartile = [0, 0, 0, 0]

        ok = True
        # loop over trials 
        for i in range(total):
        
            rng.shuffle(remaining)
            picked_idx = None

            # try to pick the next trial that fits all constraints
            for idx, cand in enumerate(remaining):

                c = cand["concept"]
                w = cand["word_shown_english"]
                is_mismatch = (cand["match_idx"] == 0)     

                # constraint 01: concept & word streaks
                cs = 1
                for pc in reversed(last_concepts):
                    if pc == c: cs += 1
                    else:
                        break 
                if cs > max_streak:
                    continue # try next item in list 

                ws = 1
                for pw in reversed(last_words):
                    if pw == w: ws += 1
                    else: break
                if ws > max_streak:
                    continue

                # constraint 03 quartile spreading: compute quartile and enforce even spread of concepts
                q = (i * 4) // total  # which quartile are we on (0-3)?
                if counts_quartile[c][q] >= target_per_quartile:
                    continue

                # constraint 05: mismatches spread evenly across quartiles 
                if is_mismatch and counts_mismatch_quartile[q] >= (MISMATCH_PER_QUARTILE + QUARTILE_TOL):
                    continue 
                

                # passed all checks
                picked_idx = idx
                break

            if picked_idx is None:
                ok = False
                break
                
            # remove item from remaining and add to trial lisr
            item = remaining.pop(picked_idx)
            schedule.append(item)

            # update trackers
            last_concepts.append(item["concept"])
            last_words.append(item["word_shown_english"])
            
            q = (i * 4) // total
            counts_quartile[item["concept"]][q] += 1

            if is_mismatch:
                counts_mismatch_quartile[q] += 1

        # schedule built
        if ok and not remaining:
            return schedule

    return []


# --- main function: build base trial list

def main(out_xlsx=base_name, seed=SEED):
    rng = random.Random(seed)
    concept_to_num = {c: i+1 for i,c in enumerate(concepts)}

    # Build base list of trials (each image once, with no constraints)
    trials = []
    for c in concepts:
        folder = folder_name_map[c]
        for ex in range(1, N_EXEMPLARS_PER_CONCEPT+1):
            img = f"{folder}\\{folder}_{ex:02d}.jpg"
            trials.append({"concept": c, "image_shown": img, "concept_num": concept_to_num[c]})
    total = len(trials)

    # get indices for mismatches 
    mismatch_index_by_concept = {c: set() for c in concepts}

    # constraint 02: mismatch quota per concept 
    for c in concepts:
        # pick which exemplar indices (1..40) for this concept will be mismatches
        idxs = list(range(1, N_EXEMPLARS_PER_CONCEPT+1))
        rng.shuffle(idxs)
        mismatch_index_by_concept[c] = set(idxs[:PER_CONCEPT_MISMATCH])

    # assign words to images to create mismatches 
    enriched = []

    # dict to store mismatches that are already used
    used_mismatches = {c: set() for c in concepts}

    for base in trials:
        c = base["concept"]
        # parse exemplar id from the file path (.._XX.jpg)
        ex_str = base["image_shown"].split("_")[-1].split(".")[0]
        ex_id = int(ex_str)
        is_mismatch = 1 if ex_id in mismatch_index_by_concept[c] else 0  # 1=mismatch here
        # match_idx: 1 for match, 0 for mismatch.
        match_idx = 0 if is_mismatch == 1 else 1

        w_en = c if match_idx == 1 else pick_nonmatching_word(c, used_mismatches, rng)
        w_de = german_map[w_en]
        enriched.append({
            **base,
            "word_shown_english": w_en,
            "word_shown_german": w_de,
            "match_idx": match_idx
        })

        n_mismatch = sum(1 for trial in enriched if trial["match_idx"] == 0)
        
    # try building schedule
    sched = build_valid_schedule(
        enriched,
        rng=rng,
        max_attempts=MAX_ATTEMPTS,
        max_streak=MAX_STREAK,
        n_exemplars_per_concept=N_EXEMPLARS_PER_CONCEPT
    )

    # add ITI and response_side mapping 
    rows = []
    correct_button_storage = deque(maxlen=MAX_STREAK_COR_KEY)

    for t in sched: # loop trials 

        iti = rng.uniform(*ITI_RANGE)

        # pick a side that doesn't violate the correct-button streak rule
        while True:
            nonmatch_pres_side = rng.choice(screen_positions)

            # correct button 
            # compute the correct response for THIS TRIAL
            if t["match_idx"] == 0:  # mismatch: nonmatch side is correct
                resp = left_key if nonmatch_pres_side == "left" else right_key
            else:                    # match: opposite side is correct
                resp = left_key if nonmatch_pres_side == "right" else right_key

            streak_counter = 1
            for past_corr in reversed(correct_button_storage): 
                if past_corr == resp: 
                    streak_counter+=1
                else: break # stop streak counter for mismatch            
            if streak_counter <= MAX_STREAK_COR_KEY:
                break 
        
        correct_button_storage.append(resp)

        # append missing rows 
        rows.append({
            "image_shown": t["image_shown"],
            "word_shown_english": t["word_shown_english"],
            "word_shown_german": t["word_shown_german"],
            "ITI_length": round(iti, 2),
            "correct_response": resp,
            "match_idx": t["match_idx"],
            "concept_num": t["concept_num"],
            "nonmatch_pres_side": nonmatch_pres_side
        })

    # to df 
    df = pd.DataFrame(rows)
        
    # export
    df.to_excel(out_xlsx, index=False)
    print(f"Wrote {out_xlsx} with {len(df)} trials, {n_mismatch} mismatches).")


if __name__ == "__main__":
    import argparse, os

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_participants", type=int, default=30)
    ap.add_argument("--out_dir", default = r"c:\sync_folder\TSRlearn\Experiment\sequences") 
    args = ap.parse_args()

    base_name = "localiser_conditions"
    os.makedirs(args.out_dir, exist_ok=True)

    for ppt in range(args.n_participants):
        ppt_id = f"{ppt:02d}"
        out_xlsx = os.path.join(args.out_dir, f"{base_name}_{ppt_id}.xlsx")
        main(out_xlsx=out_xlsx, seed=SEED + ppt)  

