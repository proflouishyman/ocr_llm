import sys
sys.path.append('/home/lhyman6/.local/lib/python3.8/site-packages')

try:
    from llava.train.train import train
    print("Module imported successfully")
except ModuleNotFoundError:
    print("Module not found")

# Rest of your training script...
import sys
print(sys.path)
