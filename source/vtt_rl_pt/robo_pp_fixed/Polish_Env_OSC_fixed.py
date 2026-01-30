# Script per creare la versione Fixed dell'ambiente
# Leggiamo il file originale e modifichiamo la sezione del trajectory manager

import re
from pathlib import Path

src_path = Path(__file__).with_name("Polish_Env_OSC.py")
with src_path.open("r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line numbers
start_line = None
end_line = None

for i, line in enumerate(lines):
    if 'Load trajectory from file' in line:
        start_line = i
    if start_line and 'self.traj_mgr.sample' in line and 'torch.arange' in line:
        end_line = i + 1
        break

print(f"Found trajectory section: lines {start_line+1} to {end_line+1}")
print("Content:")
for i in range(start_line, end_line):
    print(f"{i+1}: {lines[i]}", end='')
