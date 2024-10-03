#!/bin/bash

#SBATCH --job-name=zecheng_v6_pose   # 作业名
#SBATCH --nodes=1                          # 使用一个节点
#SBATCH --ntasks-per-node=1                # 每个节点的任务数
#SBATCH --gres=gpu:8                       # 每个节点需要8个GPU
#SBATCH --cpus-per-task=56                 # 分配给每个任务的CPU数目为该节点的CPU总数
#SBATCH --mem=800G                         # 使用该节点上所有可用内存
#SBATCH --time=infinite                    # 无限运行时间
#SBATCH --output=logs/zecheng_pose_job_id-%J.out        # 标准输出重定向到job_id-<jobid>.out文件
#SBATCH --error=logs/zecheng_pose_job_id-%J.err         # 标准错误重定向到job_id-<jobid>.err文件
#SBATCH --partition=gpu                    # 指定分区
#SBATCH --exclusive                        # 独占申请的节点

source activate zecheng

cd /public/home/zecheng/workspace/zecheng/Retrieval_Head/iclr2025/training
bash scripts/fix_pose_llama3_80k.sh

