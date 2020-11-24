#!/usr/bin/env bash

for k in 2 3 4 5 6 7 8 9 10; do
    echo $k
    python cal.py --mix uq_clean_test_bt_dedup_top10_replace_mix --split dev --score output/exp/uq_clean_test/dev/bt_dedup_top10/3B/uq_ft_margin.txt --num_bt $k --mn $k:10
done
