for sim_num in {0..20}
do
echo "submitted job simulation with seed $sim_num "
python power_motorcircle_basic.py hc $sim_num 50 100
python power_motorcircle_basic.py pt $sim_num 50 100
done
# fsl_sub -T 100 -R 32 python power_bandit4arm_combined.py pt $sim_num 70 300
# fsl_sub -T 100 -R 32 python power_bandit4arm_combined.py hc $sim_num 70 300
# done