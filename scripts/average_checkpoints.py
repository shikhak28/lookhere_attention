# save as scripts/avg_checkpoints.py
import sys, os, torch
outdir = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
files = sorted([f for f in os.listdir(outdir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth.tar')],
               key=lambda x: int(x.split('_')[-1].split('.pth')[0]))
files = files[-k:]
avg_state = {}
count = 0
for fn in files:
    ck = torch.load(os.path.join(outdir, fn), map_location='cpu')
    state = ck.get('state_dict', ck)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    if not avg_state:
        avg_state = {k: v.clone().float() for k, v in state.items()}
    else:
        for k2, v in state.items():
            avg_state[k2] += v.float()
    count += 1
for k2 in avg_state:
    avg_state[k2] /= float(count)
torch.save({'epoch':'avg_of_'+','.join(files),'state_dict':avg_state}, os.path.join(outdir, 'checkpoint_avg.pth.tar'))
print('Wrote checkpoint_avg.pth.tar averaging', files)
