import subprocess

for i in range(2, 10):
    command = ('python3 Qlearning_flie3.py --batch_number=1 -disable_until --episode_len=200 -all_map --environment=OfficeWorldTaskBCA --num_episodes=1500'+
        ' --seed=%d --result_dir=../experiments/denemeNU%d'%(i,i))
    subprocess.call(command,
        shell=True)
