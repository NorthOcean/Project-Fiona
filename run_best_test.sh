###
 # @Author: Conghao Wong
 # @Date: 2020-10-18 10:36:16
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2020-10-19 14:20:55
 # @Description: file content
### 

# # R10
# python main.py --load ./logs/20201018-100305500FRP_R10_Reverse_F0/500FRP_R10_Reverse_ --gpu 2 && # dataset 0
# python main.py --load ./logs/20201018-101331500FRP_R10_Reverse_F1/500FRP_R10_Reverse_ --gpu 2 && # dataset 1
# python main.py --load ./logs/20201018-102339500FRP_R10_Reverse_F2/500FRP_R10_Reverse_ --gpu 2 && # dataset 2
# python main.py --load ./logs/20201018-103433500FRP_R10_Reverse_F3/500FRP_R10_Reverse_ --gpu 2 && # dataset 3
# python main.py --load ./logs/20201018-104433500FRP_R10_Reverse_F4/500FRP_R10_Reverse_ --gpu 2    # dataset 4

# # R0
# python main.py --load ./logs/20201018-104440500FRP_R0_Reverse_F0/500FRP_R0_Reverse_ --gpu 2 && # dataset 0
# python main.py --load ./logs/20201018-104855500FRP_R0_Reverse_F1/500FRP_R0_Reverse_ --gpu 2 && # dataset 1
# python main.py --load ./logs/20201018-105323500FRP_R0_Reverse_F2/500FRP_R0_Reverse_ --gpu 2 && # dataset 2
# python main.py --load ./logs/20201018-105750500FRP_R0_Reverse_F3/500FRP_R0_Reverse_ --gpu 2 && # dataset 3
# python main.py --load ./logs/20201018-110140500FRP_R0_Reverse_F4/500FRP_R0_Reverse_ --gpu 2    # dataset 4

# # R6
# python main.py --load ./logs/20201018-105731500FRP_R6_Reverse_F0/500FRP_R6_Reverse_ --gpu 2 && # dataset 0
# python main.py --load ./logs/20201018-110728500FRP_R6_Reverse_F1/500FRP_R6_Reverse_ --gpu 2 && # dataset 1
# python main.py --load ./logs/20201018-111434500FRP_R6_Reverse_F2/500FRP_R6_Reverse_ --gpu 2 && # dataset 2
# python main.py --load ./logs/20201018-112505500FRP_R6_Reverse_F3/500FRP_R6_Reverse_ --gpu 2 && # dataset 3
# python main.py --load ./logs/20201018-113700500FRP_R6_Reverse_F4/500FRP_R6_Reverse_ --gpu 2    # dataset 4

# # R5
# python main.py --load ./logs/20201018-114331modelF0/model   # dataset 0 only

# Best
python main.py --load ./logs/20201018-114331modelF0/model                           --gpu 3 --sr_enable 1 --draw_results 1 --refine_epochs 10 --theta 2.5 && # dataset 0
python main.py --load ./logs/20201018-110728500FRP_R6_Reverse_F1/500FRP_R6_Reverse_ --gpu 3 --sr_enable 1 --draw_results 1 --refine_epochs 10 --theta 2.5 && # dataset 1
python main.py --load ./logs/20201018-111434500FRP_R6_Reverse_F2/500FRP_R6_Reverse_ --gpu 3 --sr_enable 1 --draw_results 1 --refine_epochs 10 --theta 2.5 && # dataset 2
python main.py --load ./logs/20201018-112505500FRP_R6_Reverse_F3/500FRP_R6_Reverse_ --gpu 3 --sr_enable 1 --draw_results 1 --refine_epochs 10 --theta 2.5 && # dataset 3
python main.py --load ./logs/20201018-104433500FRP_R10_Reverse_F4/500FRP_R10_Reverse_ --gpu 3 --sr_enable 1 --draw_results 1 --refine_epochs 10 --theta 2.5    # dataset 4