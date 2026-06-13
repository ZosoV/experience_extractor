The dataset was generated using a custom two-step Python pipeline that we built and executed directly in your workspace.

Here is exactly what scripts were used and what they did:

### 1. Generating the Prompts

We ran python -m research.generate_custom_augmented
This script was responsible for generating the core Wonderland equations. It dynamically generated 1,024 unique mathematical prompts for each dataset
size (0, 1, 2, and 3 examples).

• The Magic: We custom-coded this script to track exactly which digits were revealed in the examples. It enforces strict empirical overlap rules so the generated equations mathematically match the difficulty distribution of the original dataset.

### 2. Building the 128 Candidates

We ran python -m research.build_constrained_datasets
This script took the prompts from step 1 and generated the massive pool of mathematically valid multiple-choice answers for the student model.

• The Magic: We completely rewrote this script from scratch. It scans the query for unmapped characters (like [ or & ) and builds a massive permutation matrix across all unused digits. We heavily optimized it to use multiprocessing.Pool across all CPU cores, forcefully breaking the matrix search the exact moment it hits 500 mathematically diverse strings, picking an evenly-spaced 128 candidates from that diverse pool.
