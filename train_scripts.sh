###
 # @Author: Conghao Wong
 # @Date: 2020-10-18 09:34:33
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2020-10-18 10:36:04
 # @Description: file content
### 

cd ~/Project-Fiona && 
python main.py --gpu 3 --train_type all --model F --reverse 1 --rotate 10 --batch_size 20000 --epochs 500 --model_name 500FRP_R10_Reverse_ --lr 0.01 --train_percent 0.0 --test_set 0 && 
python main.py --gpu 3 --train_type all --model F --reverse 1 --rotate 10 --batch_size 20000 --epochs 500 --model_name 500FRP_R10_Reverse_ --lr 0.01 --train_percent 0.0 --test_set 1 && 
python main.py --gpu 3 --train_type all --model F --reverse 1 --rotate 10 --batch_size 20000 --epochs 500 --model_name 500FRP_R10_Reverse_ --lr 0.01 --train_percent 0.0 --test_set 2 && 
python main.py --gpu 3 --train_type all --model F --reverse 1 --rotate 10 --batch_size 20000 --epochs 500 --model_name 500FRP_R10_Reverse_ --lr 0.01 --train_percent 0.0 --test_set 3 && 
python main.py --gpu 3 --train_type all --model F --reverse 1 --rotate 10 --batch_size 20000 --epochs 500 --model_name 500FRP_R10_Reverse_ --lr 0.01 --train_percent 0.0 --test_set 4 # && 
# python get_all_result.py --model_name 500FRP_R10_Reverse_ --lr 0.01