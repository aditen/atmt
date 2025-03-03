import json
import shlex
import subprocess

import matplotlib.pyplot as plt
from tqdm import tqdm

evaluate = False
plot = True
if __name__ == "__main__":
    values_q = [i * 0.25 for i in range(0, 5)]
    values_t = [i * 0.1 for i in range(1, 5)]
    values = [0.15] + values_t + values_q
    values = sorted(values)
    best_beam_size = 20

    if evaluate:
        for i in tqdm(values):
            cmd = 'python translate_beam.py ' \
                  '--dicts data/en-fr/old_prepared ' \
                  '--data data/en-fr/old_prepared ' \
                  '--checkpoint-path assignments/03/baseline/checkpoints/checkpoint_last.pt ' \
                  f'--output assignments/04/baseline/alpha_{i:.2f}_raw.txt ' \
                  f'--beam-size {best_beam_size} ' \
                  f'--alpha {i:.2f}'
            if True:
                p = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                assert p.returncode == 0, "Non 0 return code for Prediction"
            p = subprocess.call(shlex.split(
                f"postprocess_cd.sh assignments/04/baseline/alpha_{i:.2f}_raw.txt assignments/04/baseline/alpha_{i:.2f}.txt en"),
                cwd="./scripts", shell=True)
            assert p == 0, "Non 0 return code for Postprocess"
            p = subprocess.run(
                f"sacrebleu data/en-fr/raw/test.en -i assignments/04/baseline/alpha_{i:.2f}.txt > assignments/04/baseline/alpha_{i:.2f}_scores.json",
                shell=True)
            assert p.returncode == 0, "Non 0 return code for Sacrebleu"

    if plot:
        bleu_scores = []
        brev_pens = []
        fig, ax1 = plt.subplots()
        for val in values:
            filename = f'assignments/04/baseline/alpha_{val:.2f}_scores.json'
            json_data = json.loads(open(filename, 'r', encoding='utf-8').read())
            bleu_scores.append(json_data['score'])
            bp_start_index = json_data['verbose_score'].index("BP = ") + len("BP = ")
            bp_end_index = json_data['verbose_score'].index(" ", bp_start_index)
            brev_pens.append(float(json_data['verbose_score'][bp_start_index:bp_end_index]))
        ax1.plot(values, bleu_scores, "-o", color="blue")
        plt.title('BLEU Score (blue) and Brevity Penalty (red) for different Alphas')
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('BLEU Score')
        ax1.set_ylim((0, max(bleu_scores) + 1))

        ax2 = ax1.twinx()
        ax2.plot(values, brev_pens, "-^", color="red")
        ax1.set_xlabel('Alpha')
        ax2.set_ylabel("Brevity Penalty")
        ax2.set_ylim((0, 1.05))
        plt.show()
        fig.savefig("assignments/04/baseline/alphas_plot.png")
