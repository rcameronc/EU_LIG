#!/bin/bash

# Ice model
for name in d6g_h6g_ # glac1d_ # d6g_h6g_ # glac1d_
do

#Earth model
for lith in l96C # l71C
do
 
for lm in 3 5 7 8 9 10 15 20 30 40 50
do

for um in p2 p3 p4 p5
do

for tmax in 12000 # 15000
do

for tmin in 100 # 5990 1990
do

for place in northsea_uk_tight # arctic # # europe # norway # 
do

# put together file name
fileName="execute_${name}${lith}_${um}_${lm}_${tmax}_${tmin}_${place}"
fileName_run="run_${name}${lith}_${um}_${lm}_${tmax}_${tmin}_${place}.sh"
fileName_out="out_${name}${lith}_${um}_${lm}_${tmax}_${tmin}_${place}.out"
run_name="${name}${lith}_${um}_${lm}_${tmax}_${tmin}_${place}";


## create this folder in the same place as this file
mkdir run_eu

# go to run folder
cd run_eu

# write an execute script that passes parameters on to execute script
rm $fileName_run

# Open file descriptor (fd) 4 for read/write on a text file.
exec 4<> $fileName_run

    # Let's print some text to fd 3
    echo "cd .." >&4
    echo "source activate gpflow6_0" >&4
    echo "python -m memory_profiler eu_lig_it.py --mod $name --lith $lith --um $um --lm $lm --tmax $tmax --tmin $tmin --place $place" >&4
    echo "exit" >&4

# Close fd 4
exec 4>&-

## create this folder in the same directory as this file

cd ..
mkdir execute_eu

# go to execute folder
cd execute_eu
# rm $fileName
# write a submit script that passes parameters on to execute script

   # Open file descriptor (fd) 3 for read/write on a text file.
    exec 3<> $fileName

    # Let's print some text to fd 3
    echo "#!/bin/bash" >&3
    echo "#SBATCH -o $fileName_out" >&3
    echo "#SBATCH -A jalab" >&3
    echo "#SBATCH -J $run_name" >&3
    echo "#SBATCH --gres=gpu:1" >&3
#     echo "#SBATCH --mem-per-cpu=125gb" >&3
    echo "#SBATCH --time=4:30:00" >&3
    echo "#SBATCH --mail-type=ALL"  >&3  # specify what kind of emails you want to get
    echo "#SBATCH --mail-user=rcreel@ldeo.columbia.edu" >&3  # specify email address"
    echo " " >&3
    echo "module load singularity" >&3
    echo "module load cuda80/toolkit" >&3
    echo "cd ../run_eu/" >&3
    echo "singularity exec --nv /rigel/jalab/users/rcc2167/gpflow-tensorflow-rcc2167.simg bash ${fileName_run}" >&3


    # Close fd 3
    exec 3>&-

# submit execute file

eval "sbatch $fileName"
echo "sbatch $fileName"
#cd ../code

# go back to start
cd ..

done
done
done
done
done
done
done
