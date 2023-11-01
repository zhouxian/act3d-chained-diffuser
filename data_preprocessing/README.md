# Data Generation

## 1 - HiveFormer Data Generation
```
root=/home/theophile_gervet_gmail_com
data_dir=$root/datasets/raw
output_dir=$root/datasets/packaged
train_dir=74_hiveformer_tasks_train
val_dir=74_hiveformer_tasks_val
train_episodes_per_task=100
val_episodes_per_task=100
image_size="256,256"
task_file=tasks/hiveformer_74_tasks.csv

nohup sudo X &
export DISPLAY=:0.0
```

### A - Generate raw train and val data
```
cd $root/hiveformer/RLBench/tools

processes=3
python dataset_generator.py \
    --save_path=$data_dir/$train_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=$processes; python dataset_generator.py \
    --save_path=$data_dir/$val_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=$processes
```

### B - Preprocess train and val data
```
cd $root/hiveformer
for task in $(cat $task_file | tr '\n' ' '); do
    for split_dir in $train_dir $val_dir; do
        python -m data_preprocessing.data_gen \
            --data_dir=$data_dir/$split_dir \
            --output=$output_dir/$split_dir \
            --image_size=$image_size \
            --max_variations=1 \
            --tasks=$task
    done
done
```

## 1 - PerAct Data Generation
```
root=/home/theophile_gervet_gmail_com
data_dir=$root/datasets/raw
output_dir=$root/datasets/packaged
train_dir=18_peract_tasks_train
val_dir=18_peract_tasks_val
train_episodes_per_task=100
val_episodes_per_task=100
image_size="256,256"
task_file=tasks/peract_18_tasks.csv
```

### A - Generate raw train and val data
```
cd $root/hiveformer/RLBench/tools

processes=3
python dataset_generator.py \
    --save_path=$data_dir/$train_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=-1 \
    --offset=0 \
    --processes=$processes; python dataset_generator.py \
    --save_path=$data_dir/$val_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=-1 \
    --offset=0 \
    --processes=$processes
```

### B - Preprocess train and val data
```
cd $root/hiveformer
for task in $(cat $task_file | tr '\n' ' '); do
    for split_dir in $train_dir $val_dir; do
        python -m data_preprocessing.data_gen \
            --data_dir=$data_dir/$split_dir \
            --output=$output_dir/$split_dir \
            --image_size=$image_size \
            --max_variations=60 \
            --tasks=$task
    done
done
```

## 3 - Preprocess Instructions for Both Datasets
```
root=/home/theophile_gervet_gmail_com
cd $root/hiveformer

task_file=tasks/82_all_tasks.csv
python -m data_preprocessing.preprocess_instructions \
    --tasks $(cat $task_file | tr '\n' ' ') \
    --output instructions.pkl \
    --variations {0..199} \
    --annotations data_preprocessing/annotations.json
```

## 4 - Compute Workspace Bounds for Both Datasets
```
root=/home/theophile_gervet_gmail_com
cd $root/hiveformer

output_dir=$root/datasets/packaged
train_dir=74_hiveformer_tasks_train
task_file=tasks/hiveformer_74_tasks.csv
python -m data_preprocessing.compute_workspace_bounds \
    --dataset $output_dir/$train_dir \
    --out_file 74_hiveformer_tasks_location_bounds.json \
    --tasks $(cat $task_file | tr '\n' ' ')

output_dir=$root/datasets/packaged
train_dir=18_peract_tasks_train
task_file=tasks/peract_18_tasks.csv
python -m data_preprocessing.compute_workspace_bounds \
    --dataset $output_dir/$train_dir \
    --out_file 18_peract_tasks_location_bounds.json \
    --variations {0..199} \
    --tasks $(cat $task_file | tr '\n' ' ')
```
