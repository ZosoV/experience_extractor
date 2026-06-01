# Single Equation Generator

## Description

This is the spec with all the guidelines to create a generator of solution based solely on one equation.

Keep the code of this generator in src/single_eq_generator

The objective is to generate one or k solutions per each symbolic equation, when this equation is unconstrained to other examples.

The resulting database will have four columns

- symbolic equation (str) the whole equation lhs = rhs
- lhs (str): the lhs
- answer (str): the rhs
- reasoning (str): the reasoning of the solution, operation mapping, digit mapping, base, and solver strategy
- generated_prompt (str): a prompt with the same header of Alice in the Wonderland but without examples

NOTE: When the symbolic equation corresponds to a pure concat on strings (concat forward: l0l1r0r1 or concat reverse: r0r1l0l1 on strings), just skip that equation. We don't need to pretrain the NN on these cases for now.

## Execution Flow

0. Set a configuration file with some hyperparameters, e.g: num_of_solution per sample
1. The entry prompt and answer is feed and the symbolic equations are extracted including the last lhs = answer.
2. Each symbolic equation (e.g. `!*\& = '@'{ ), including the one with answer, are divided on relevant sections l0, l1, operator, r0, r1 = rhs
3. Choose one symbolic eq
4. Check whether the equation can be solved by pure concat or not (concat forward: l0l1r0r1 or concat reverse: r0r1l0l1 on strings). 
        - If it can be solved by pure concat, skip
        - else continue
5. Choose randomly a solver strategy: standard or little_endian
    - standard: `AB` → `mapping[A] × base + mapping[B]`, A: can be l0 or r0, and B can be l1 or r1
    - little_endian: `AB` → `mapping[B] × base + mapping[A]`, A: can be l0 or r0, and B can be l1 or r1
6. Set the base = 10 (Fixed for now)
6. Chose randomnly an operation from the set of all possible operations defined on OPERATIONS in the script operations.py.
7. Chose