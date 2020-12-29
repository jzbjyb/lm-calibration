#!/usr/bin/env bash

for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    echo $k
    python cal.py --mix uq_clean_test_bt_dedup_top20_replace_mix --split dev --score output/exp/uq_clean_test/dev_nolennorm/bt_dedup_top20/3B/uq_ft_margin.txt --num_bt $k --mn $k:20
done
