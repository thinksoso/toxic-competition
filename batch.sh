#!/bin/bash
#SBATCH -J toxic                            # 作业名为 test
#SBATCH -o toxic_version2.out                         # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH -t 40:00:00                           # 任务运行的最长时间为 20 小时
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1
source ~/.bashrc

# 设置运行环境
#pyenv activate devel_cu102_1.7.1
conda activate base
# 输入要执行的命令，例如 ./hello 或 python test.py 等
#python run.py                    # 执行命令
python run.py 
