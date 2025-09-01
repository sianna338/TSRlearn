
import random
import pandas as pd
from collections import defaultdict, deque
from typing import List, Dict

# --- settings ---
SEED = 42 # 
N_EXEMPLARS_PER_CONCEPT = 40
SEQUENCE_RUNS = {"A": 30, "B": 20, "C": 10} # how many runs per sequence overall 
BLOCK_ORDER = [("A", 10), ("A", 10), ("B", 10), ("A", 10), ("B", 10), ("C", 10)] # sequence, n_repetitions
POSITION_KEYS = {"left": "y", "center": "b", "right": "r"} # map image positions to response keys 
concepts = [
    "berry","bicycle","bird","box","bug","car","chair","coffee","dog","face","fish",
    "guitar","hammer","hand","house","jacket","pencil","phone","pizza","plane","tree"
]
base_name = "main_conditions" # root filename without ppt index
n_participants = 30

# calculate total trial number for each sequence (given block design)
total_trials_per_seq = defaultdict(int)

for label, count in BLOCK_ORDER:
    total_trials_per_seq[label] += count * 6

# --- constraints for generating the condition file ---

# Constraint 1: max same position streak for images from the same condition
MAX_SAME_POSITION_STREAK = 2 

# Constraint 2: min distance (number of images) between distactor_01 and current image 
MIN_DISTANCE_DIS_CUR = 2 # at least one image in between 

# Constraint 3: each condition (correct, dist_01, dist_02) appears ~equally in each position 
total_trials = (SEQUENCE_RUNS["A"] + SEQUENCE_RUNS["B"] + SEQUENCE_RUNS["C"]) * 6
appearances_per_pos = total_trials // 3  # ~120 for 360 trials

pos_targets = {
    "correct": {"left": appearances_per_pos, "center": appearances_per_pos, "right": total_trials - 2*appearances_per_pos},
    "dist01":  {"left": appearances_per_pos, "center": appearances_per_pos, "right": total_trials - 2*appearances_per_pos},
    "dist02":  {"left": appearances_per_pos, "center": appearances_per_pos, "right": total_trials - 2*appearances_per_pos},
}

# Constraint 4: dist_02 has equal probability of coming from both of the other two sequences  

# Constraint 5: each eligible same-sequence concept for dist_01 is sampled fairly equal across learning route

# Constraint 6: no same concept appearing as prompt (current) on trial t and as dist_01 or dist_02 on trial t+1 
PROMPT_TO_DIST_LAG = 1 # at least one other concept in between, this cannot be 2 apparently

# Constraint 7: no same concept appearing as distractor (dist_01 or dist_02) on trial t and as dist_01 or dist_02 on trial t+1   
DIST_TO_DIST_LAG = 1 # at least one other concept in between 

# --- max attempts to find sequence that fullfills constraint ---
MAX_SIM_ATTEMPTS = 5000

# create dict with concept names
folder_name_map = {c: c for c in concepts}

# --- function that picks dist_02 stimulus on each trial 
def pick_dist02_concept(rng, learn_label, seq_map, dist02_quota,
                        recent_prompts, recent_dists,
                        prompt_to_dist_lag, dist_to_dist_lag):
    """Return (dist02_concept, source_seq) or (None, None) if impossible."""

    # prepare candidate sources with remaining quota
    candidates = []

    # loop over both source sequences and number of dist_02 items
    for src, remaining in dist02_quota[learn_label].items():
        if remaining <= 0:
            continue
        # apply lag filters
        valids = [c for c in seq_map[src] # get concepts for this sequence
                  if (prompt_to_dist_lag == 0 or c not in recent_prompts)
                  and (dist_to_dist_lag == 0 or c not in recent_dists)]
        if valids:
            # create list with all candidate dist02s 
            candidates.append((src, remaining, valids))

    if not candidates:
        return None, None

    # weights depending on how many dist02s from this sequence are left 
    weights = [rem for (_, rem, _) in candidates]
    idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
    src, _, valids = candidates[idx]
    concept = rng.choice(valids)
    dist02_quota[learn_label][src] -= 1
    return concept, src

    # --- old function: ensure that distractor_02 comes from the two possible sequences with equal possibility ---
    # this does not have the weights but just randomly chooses from the list of dist02 candiates 
    # def build_balanced_sources(learning_label: str, n_trials: int, rng: random.Random) -> List[str]:
        # others = [lab for lab in ["A","B","C"] if lab != learning_label] # get the two sequences
        # half = n_trials // 2
        # remainder = n_trials % 2

        # list of possible options for dist_02
        # sources = [others[0]] * half + [others[1]] * half 
        # if remainder:
        #     sources.append(rng.choice(others))
        # rng.shuffle(sources)
        # return sources

# -- function to find image sources --- 
def file_path_for(concept: str, exemplar_id: int) -> str:
    folder = folder_name_map[concept]
    return f"{folder}//{folder}_{exemplar_id:02d}.jpg"

# -- function to match three choice options to positions --- 
def sample_positions_with_streaks(rng, last_positions, pos_tally, pos_targets):

    # possible position combinations 
    perms = [
        ("left","center","right"),("left","right","center"),
        ("center","left","right"),("center","right","left"),
        ("right","left","center"),("right","center","left"),
    ]
    rng.shuffle(perms)

    best_perm = None
    best_score = -1e9

    for correct_pos, dist01_pos, dist02_pos in perms:
        ok = True
        score = 0.0
        for cond, pos in [("correct", correct_pos), ("dist01", dist01_pos), ("dist02", dist02_pos)]:
            # make sure max_pos_streak is not exceeded 
            streak = 0
            for p in reversed(last_positions[cond]):
                if p == pos: streak += 1
                else: break
            if streak + 1 > MAX_SAME_POSITION_STREAK:
                ok = False
                break

            # now make sure each cond appears in each position the same amount of timees 
            used = pos_tally[cond][pos]
            target = pos_targets[cond][pos]
            deficit = target - used
            score += deficit  # larger deficit = more desirable
        if ok and score > best_score:
            best_score = score

            best_perm = (correct_pos, dist01_pos, dist02_pos)

    return best_perm 


def init_dist02_quota():
    quota = {}
    for learn in ["A","B","C"]:
        others = [x for x in ["A","B","C"] if x != learn]
        tot = total_trials_per_seq[learn]
        half = tot // 2
        quota[learn] = {others[0]: half, others[1]: tot - half}  # exact 50/50 (±1 if odd)
    return quota


# --- helper: choose the least used dist_01 ---
def pick_dist01_fair(rng, seq_label, t, dist01_usage):

    #t is index of trial within run of 6 trials 
    elig = [j for j in range(7) if (abs(j - t) >= MIN_DISTANCE_DIS_CUR and j not in (t, t+1))]
    if not elig: 
        return None
    
    # choose the least-used j (do we need this as constraint or can we just do random choice??)
    rng.shuffle(elig)
    j = min(elig, key=lambda k: dist01_usage[seq_label][t][k])
    dist01_usage[seq_label][t][j] += 1
    return j

# --- main function: build trial sequence ---
def attempt_build(rng, seq_map, sequence_run_exemplars, seed=SEED):

    rows_local = []

    # give each concept a number to send triggers in the experiment 
    concept_to_trig = {c: (i+1) for i, c in enumerate(concepts)}

    # store for seq A, how many dist_02s come from seq B and seq C, and so on 
    dist02_quota_local = init_dist02_quota()  # set new on each attempt

    # contraints 05 (count how often each of the 7 sequence concepts are used as dist_01 on learning trials)
    dist01_usage = {lab: {i: defaultdict(int) for i in range(7)} for lab in ["A","B","C"]}
    
    # to store posititions on last trials
    last_positions_local = {"correct": deque(maxlen=MAX_SAME_POSITION_STREAK),
                            "dist01": deque(maxlen=MAX_SAME_POSITION_STREAK),
                            "dist02": deque(maxlen=MAX_SAME_POSITION_STREAK)}
    
    # to store count of positions across whole task 
    pos_conditions_global = {"correct": {"left":0,"center":0,"right":0},
                                "dist01":  {"left":0,"center":0,"right":0},
                                "dist02":  {"left":0,"center":0,"right":0}}
    
    # to store recent concepts to avoid clustering 
    recent_prompts = deque(maxlen=PROMPT_TO_DIST_LAG-1 if PROMPT_TO_DIST_LAG>0 else 0)
    recent_dists   = deque(maxlen=DIST_TO_DIST_LAG-1 if DIST_TO_DIST_LAG>0 else 0)

    # to keep track how many routes / runs per sequence were already created 
    run_index_local = {"A": 0, "B": 0, "C": 0}

    for label, n_runs in BLOCK_ORDER:

        # get concepts from this sequence
        seq_concepts = seq_map[label]

        # loop over runs 
        for _ in range(n_runs):
            run_idx = run_index_local[label]

            # get examplars for the run 
            exemplars_for_run = sequence_run_exemplars[label][run_idx]
            run_index_local[label] += 1

            # loop over 6 trials within run 
            for t in range(6):
                curr_concept = seq_concepts[t]
                next_concept = seq_concepts[t+1]

                # pick dist_01 (constraint 02 & 05)
                j_idx = pick_dist01_fair(rng, label, t, dist01_usage)
                if j_idx is None:
                    return None
                dist01_concept = seq_concepts[j_idx]

                # Constraint 06 & 07: lag from recent prompts and recent distractors
                if PROMPT_TO_DIST_LAG > 0 and dist01_concept in recent_prompts:
                    return None
                if DIST_TO_DIST_LAG > 0 and dist01_concept in recent_dists:
                    return None

                # pick dist_02 (constraint 04)
                dist02_concept, source_seq = pick_dist02_concept(
                    rng=rng,
                    learn_label=label,
                    seq_map=seq_map,
                    dist02_quota=dist02_quota_local,
                    recent_prompts=recent_prompts,
                    recent_dists=recent_dists,
                    prompt_to_dist_lag=PROMPT_TO_DIST_LAG,
                    dist_to_dist_lag=DIST_TO_DIST_LAG
                )
                if dist02_concept is None:
                    # no lag-valid concept from either source with remaining quota
                    return None

                # get images for examplars on this run 
                prompt_file = file_path_for(curr_concept, exemplars_for_run[curr_concept])
                correct_file = file_path_for(next_concept, exemplars_for_run[next_concept])
                dist01_file = file_path_for(dist01_concept, exemplars_for_run[dist01_concept])

                # dist02 is the only stimulus that is picked randomly from the pool of exemplars
                # and not determined before this function
                dist02_exemplar = rng.randint(1, N_EXEMPLARS_PER_CONCEPT)
                dist02_file = file_path_for(dist02_concept, dist02_exemplar)

                # constraint 03 
                positions = sample_positions_with_streaks(rng, last_positions_local, pos_conditions_global, pos_targets)
                if positions is None:
                    return None
                correct_pos, dist01_pos, dist02_pos = positions

                # store positions
                last_positions_local["correct"].append(correct_pos)
                last_positions_local["dist01"].append(dist01_pos)
                last_positions_local["dist02"].append(dist02_pos)

                # update counters of positions
                pos_conditions_global["correct"][correct_pos] += 1
                pos_conditions_global["dist01"][dist01_pos]   += 1
                pos_conditions_global["dist02"][dist02_pos]   += 1

                # update recently used concepts 
                recent_prompts.append(curr_concept)
                recent_dists.append(dist01_concept)
                recent_dists.append(dist02_concept)

                if j_idx is None:
                    print("Failed: no eligible dist01")
                    return None
                if dist02_concept is None:
                    print("Failed: ran out of dist02 sources")
                    return None
                if dist01_concept in recent_prompts:
                        print("Failed: dist_01 to prompt lag")
                        return None
                if dist01_concept in recent_dists:
                    print("Failed: dist to dist lag")
                    return None
                if positions is None:
                    print("Failed: no valid positions")
                    return None

                # get correct answer key 
                correct_ans = POSITION_KEYS[correct_pos]

                # rows for excel file 
                row = {
                    "promptFile": prompt_file,
                    "correctFile": correct_file,
                    "dist_01File": dist01_file,
                    "dist_02File": dist02_file,
                    "correct_pos": correct_pos,
                    "dist01_pos": dist01_pos,
                    "dist02_pos": dist02_pos,
                    "promptTrig": concept_to_trig[curr_concept],
                    "correctTrig": concept_to_trig[next_concept],
                    "dist_01Trig": concept_to_trig[dist01_concept],
                    "dist_02Trig": concept_to_trig[dist02_concept],
                    "correct_ans": correct_ans,
                    "learningSeq": label,
                    "currPosInSeq": t+1,
                    "dist02SourceSeq": source_seq,
                }
                rows_local.append(row)
    print("succesful attempt")
    return rows_local


# --- function: build sequences and decide when to use which exemplar ---
def main(out_xlsx=base_name, seed=SEED):

    # Partition into 3 sequences of 7 (random)
    concept_indices = list(range(len(concepts)))

    rng = random.Random(seed)
    rng.shuffle(concept_indices)
    seq_map = {}
    for label, idxs in zip(["A","B","C"], [concept_indices[i*7:(i+1)*7] for i in range(3)]):
        # define 3 sequences of 7 concepts each and store in dict
        seq_map[label] = [concepts[i] for i in idxs]

    # dict to store used exemplars for concepts 
    concept_exemplar_used = {c: set() for c in concepts}

    # dict with sequence label and then examplar number for each concept 
    sequence_run_exemplars: Dict[str, List[Dict[str, int]]] = {"A": [], "B": [], "C": []}

    # loop over sequences 
    for label in ["A","B","C"]:

        # get number of runs 
        runs = SEQUENCE_RUNS[label]

        # get all concepts in this sequence 
        seq_concepts = seq_map[label]

        # loop over runs 
        for _ in range(runs):

            # put chosen exemplars for this run for each concept in here 
            run_assignment = {}

            # loop over concepts 
            for c in seq_concepts:

                # get indices of exemplars for this concept that are not yet used
                available = [i for i in range(1, N_EXEMPLARS_PER_CONCEPT+1) if i not in concept_exemplar_used[c]]

                if not available:
                    raise RuntimeError(f"Ran out of exemplars for concept {c} in sequence {label}. Increase N_EXEMPLARS_PER_CONCEPT or lower runs.")
                
                # choose examplars for this concept 
                chosen = rng.choice(available)

                run_assignment[c] = chosen

                concept_exemplar_used[c].add(chosen)
            # add chosen exemplars for each concept to the current sequence     
            sequence_run_exemplars[label].append(run_assignment)


    # --- attempt to build sequences that meet all constraints ---
    rows = None
    for attempt in range(1, MAX_SIM_ATTEMPTS+1):

        print(f"Attempt {attempt}")

        rows = attempt_build(rng, seq_map, sequence_run_exemplars)
        if rows is not None: # if valid trial sequence found 
            break
    if rows is None:
        raise RuntimeError("Failed to satisfy constraints. Try increasing MAX_SIM_ATTEMPTS or relaxing constraints.")

    df = pd.DataFrame(rows)
    base_cols = ["promptFile","correctFile","dist_01File","dist_02File",
                 "correct_pos","dist01_pos","dist02_pos",
                 "promptTrig","correctTrig","dist_01Trig","dist_02Trig","correct_ans"]
    extra_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + extra_cols]

    # --- Sanity checks ---

    # number of trials
    total_rows_expected = (SEQUENCE_RUNS["A"] + SEQUENCE_RUNS["B"] + SEQUENCE_RUNS["C"]) * 6
    assert len(df) == total_rows_expected, f"Unexpected number of rows: {len(df)} vs {total_rows_expected}"

    # is cond 1 met?
    for cond in ["correct_pos","dist01_pos","dist02_pos"]:
        streak = 1
        last = None
        for pos in df[cond]:
            if pos == last:
                streak += 1
                if streak > MAX_SAME_POSITION_STREAK:
                    raise AssertionError(f"Position streak violated for {cond}: > {MAX_SAME_POSITION_STREAK}")
            else:
                streak = 1
                last = pos

    # is cond 4 met? 
    # (allow small tolerance when lags are active) ---
    for seq in ["A","B","C"]:
        subset = df[df["learningSeq"] == seq]
        counts = subset["dist02SourceSeq"].value_counts().to_dict()
        others = [source for source in ["A","B","C"] if source != seq]
        diff = abs(counts.get(others[0],0) - counts.get(others[1],0))
        if diff > 1:
            raise AssertionError(f"dist_02 balancing off for learning {seq}: {counts}")


        df.to_excel(out_xlsx, index=False)
        print(f"Wrote {out_xlsx} with {len(df)} rows.")

if __name__ == "__main__":

    import argparse, os

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_participants", type=int, default=30)
    ap.add_argument("--out_dir", default="c:\sync_folder\Study 1 - MEG replay\Experiment\learning-replay-MEG-task\sequences")
    args = ap.parse_args()

    base_name = "main_conditions"
    os.makedirs(args.out_dir, exist_ok=True)

    for ppt in range(args.n_participants):
        ppt_id = f"{ppt:02d}"
        out_xlsx = os.path.join(args.out_dir, f"{base_name}_{ppt_id}.xlsx")
        main(out_xlsx=out_xlsx, seed=SEED + ppt) 

