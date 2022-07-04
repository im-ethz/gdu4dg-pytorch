lambda_sparse=0.001
lambda_OLS=0.001
lambda_orth=0

# End-to-end experiments (E2E)
for test_source in "mnistm" "syn" "svhn" "usps" "mnist"
do
  for similarity in "cosine_similarity" "MMD" "projected"
  do
    for fine_tune in "False"
    do
      for run in 0 1 2 3 4 5 6 7 8 9
      do
            bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python experiments/digits5/main.py --similarity $similarity --TARGET_DOMAIN $test_source --fine_tune ${fine_tune} --lambda_orth ${lambda_orth} --lambda_sparse $lambda_sparse --lambda_OLS $lambda_OLS --run $run >& tenruns_${test_source}_${similarity}_${fine_tune}_${run}.out"
      done
    done
  done
done

# Fine-tuning experiments (FT) as method should be set to None and fint_tune to "True" _simultaneoulsy_ 
for test_source in "mnistm" "syn" "svhn" "usps" "mnist"
do
  for similarity in "None"
  do
    for fine_tune in "True"
    do
      for run in 0 1 2 3 4 5 6 7 8 9
      do
            bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python experiments/digits5/main.py --similarity $similarity --TARGET_DOMAIN $test_source --fine_tune ${fine_tune} --lambda_orth ${lambda_orth} --lambda_sparse $lambda_sparse --lambda_OLS $lambda_OLS --run $run >& tenruns_${test_source}_${similarity}_${fine_tune}_${run}.out"
      done
    done
  done
done
