import subprocess
import time


print("\nSetting up environment...")
start_time = time.time()

all_process = [
    ['pip', 'install', '-r', 'requirements.txt'],
    ['pip', 'install', 'torch==1.12.1+cu116', 'torchvision==0.13.1+cu116', '--extra-index-url', 'https://download.pytorch.org/whl/cu116'],
]

for process in all_process:
    running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')

end_time = time.time()
print(f"\nEnvironment set up in {end_time-start_time:.0f} seconds")