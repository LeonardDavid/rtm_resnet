#!/bin/bash
##########################################################################################
#   LDB
#   
#   automation script for running RTM simulation
# 
#   args:
#   $1  NN_MODEL:   FMNIST  CIFAR   RESNET
#   $2  LOOPS:      Number of inference iterations
#   $3  GPU:        GPU to be used (0, 1)
#
##########################################################################################

NN_MODEL="$1"                       # FMNIST    CIFAR   RESNET

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
results_dir="RTM_results/$NN_MODEL/$timestamp"
output_dir="$results_dir/outputs"
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    else
        echo "Directory $output_dir already exists."
    fi
else
    echo "Directory $results_dir already exists."
fi


if [ "$NN_MODEL" = "FMNIST" ]
then
    MODEL="VGG3"
    DATASET="FMNIST"
    declare -a PROTECT_LAYERS=(1 1 1 1)
    MODEL_PATH="model_fmnist9108.pt"
elif [ "$NN_MODEL" = "CIFAR" ]
then
    MODEL="VGG7"
    DATASET="CIFAR10"
    declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 1)
    MODEL_PATH="model_cifar8582.pt"
elif [ "$NN_MODEL" = "RESNET" ]
then 
    MODEL="ResNet"
    DATASET="IMAGENETTE"
    declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
    MODEL_PATH="model_resnet7694.pt"
else
    echo -e "\n\033[0;31m$NN_MODEL not supported, check spelling & available models: FMNIST, CIFAR, RESNET\033[0m\n"
    exit
fi

## default params
BATCH_SIZE=256
EPOCHS=1
LR=0.001
STEP_SIZE=25

## params for rtm testing
TEST_ERROR=1
TEST_RTM=1
LOOPS=$2
GPU=$3
# declare -a PERRORS=(0.1)
# declare -a PERRORS=(0.1 0.01 0.001 0.0001)
# declare -a PERRORS=(0.001 0.0001 0.00001 0.000001)
# declare -a PERRORS=(0.0000455 0.0000995 0.000207 0.000376 0.000594 0.000843 0.0011)


# declare -a PERRORS=(0.0001 0.0000455)
# declare -a PERRORS=(0.0001 0.0000455 0.00001 0.000001)

# declare -a PERRORS=(0.0001)
# declare -a PERRORS=(0.0000455)
# declare -a PERRORS=(0.00001)
declare -a PERRORS=(0.000001)

for p in "${PERRORS[@]}"
do
    echo -e "\n\033[0;32mRunning $NN_MODEL for $LOOPS loops with error: $p\033[0m\n"
    
    declare -a list
    
    for layer in "${!PROTECT_LAYERS[@]}"    
    do
        let "L=layer+1"
        echo -e "\n\033[0;33mUNprotected Layer: $L\033[0m\n"

        PROTECT_LAYERS[$layer]=0
        # echo "${PROTECT_LAYERS[@]}"
        
        output_dir_L="$output_dir/$L"
        if [ ! -d "$output_dir_L" ]; then
            mkdir -p "$output_dir_L"
        else
            echo "Directory $output_dir_L already exists."
        fi
        output_file="$output_dir_L/output_${DATASET}_$L-$LOOPS-$p.txt"

        python run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --test_rtm=${TEST_RTM} --gpu-num=$GPU --protect_layers ${PROTECT_LAYERS[@]} | tee "$output_file"
        
        penultimate_line=$(tail -n 2 "$output_file" | head -n 1)
        # Remove square brackets and split values
        values=$(echo "$penultimate_line" | tr -d '[]')

        list+=("$values")
        
        # echo $list
        
        python plot_new_table.py ${output_file} ${results_dir} ${NN_MODEL} ${LOOPS} ${p} ${L}

        PROTECT_LAYERS[$layer]=1
    done

    csv_file="$output_dir/table_$p.csv"

    for value in "${list[@]}"
    do
        echo "${value[@]}" >> "$csv_file"
    done

    unset list
    
done





# Check if an input file is provided as an argument
# if [[ $OUTPUT_FILE -ne 1 ]]; then
#     echo "Usage: $0 <input_file>"
#     exit 1
# fi

# # Check if the input file exists
# if [[ ! -f $OUTPUT_FILE ]]; then
#     echo "Input file not found: $OUTPUT_FILE"
#     exit 1
# fi

# # Read the penultimate line from the input file
# penultimate_line=$(tail -n 2 $OUTPUT_FILE | head -n 1)

# # Convert the penultimate line into an array of double values
# read -ra double_array <<< "$penultimate_line"

# # Print the double values in the array
# for value in "${double_array[@]}"; do
#     echo "$value"
# done

