for dataset in "civilcomments" #"fmow" #"camelyon17" #"poverty" "fmow" "iwildcam" "civilcomments" "ogb-molpcba" "rxrx1" "amazon" "camelyon17"
do
  for seed in 0 1 2 #3 4 5 6 7 8 9 # A" "B" "C" "D" "E"#0 1 2 3 4 5 6 7 8 9
  do
        python /local/home/sfoell/GitHub/gdu-pytorch/experiments/wilds/cluster-analysis.py --device "0" --seed $seed --dataset $dataset --algorithm "GDU" --download "True" --root_dir '/local/home/sfoell/GitHub/gdu-pytorch/experiments/wilds/data'
        #python /local/home/sfoell/GitHub/gdu-pytorch/experiments/wilds/cluster-analysis.py --device "1"  --dataset $dataset --dataset_kwargs "fold=$seed" --algorithm "GDU" --root_dir '/local/home/sfoell/GitHub/gdu-pytorch/experiments/wilds/data'
	done
done

