<digit_mapping_1>
Let's assign priority using a logic to analyze the most-constrained first.

Score(sym) = occ(*) × W*  +  occ(-) × W-
  where occ(op) = total occurrences of the symbol across all operand and result slots
        in equations belonging to operator op's group
  and   W = 1 + #candidates for that operator group

At this point: `*` has 4 candidates → W* = 1+4 = 5; `-` has 8 candidates → W- = 1+8 = 9.

sym   occ(*)  occ(-)   score
'       5       1      5×5 + 1×9 = 34
`       3       1      3×5 + 1×9 = 24
!       3       1      3×5 + 1×9 = 24   [tie with ` → ` first by earlier appearance]
[       3       0      3×5 + 0×9 = 15
\       1       1      1×5 + 1×9 = 14
{       2       0      2×5 + 0×9 = 10
@       2       0      2×5 + 0×9 = 10   [tie with { → { first by earlier appearance]
"       1       0      1×5 + 0×9 =  5
>       1       0      1×5 + 0×9 =  5
&       1       0      1×5 + 0×9 =  5

Assigning a random mapping by priority:

- from {0,1,2,3,4,5,6,7,8,9} picking randomly: ' = 2,
- from {0,1,3,4,5,6,7,8,9} picking randomly: ` = 5,
- from {0,1,3,4,6,7,8,9} picking randomly: ! = 6,
- from {0,1,3,4,7,8,9} picking randomly: [ = 4,
- from {0,1,3,7,8,9} picking randomly: \ = 3,
- from {0,1,7,8,9} picking randomly: { = 8,
- from {0,1,7,9} picking randomly: @ = 1,
- from {0,7,9} picking randomly: " = 7,
- from {0,9} picking randomly: > = 0,
- from {9} picking: & = 9

</digit_mapping_1>