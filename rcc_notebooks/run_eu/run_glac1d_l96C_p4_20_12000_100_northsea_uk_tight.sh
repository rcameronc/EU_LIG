cd ..
source activate gpflow6_0
python -m memory_profiler eu_lig_it.py --mod glac1d_ --lith l96C --um p4 --lm 20 --tmax 12000 --tmin 100 --place northsea_uk_tight
exit
