import json
import shlex
import subprocess

import matplotlib.pyplot as plt
from tqdm import tqdm

evaluate = False
plot = True
best_alpha = 0.1
best_beam = 20

if __name__ == "__main__":
    values = [0.001, 0.005, 0.01, 0.1, 0.2]
    values = sorted(values)

    if evaluate:
        for i in tqdm(values):
            cmd = 'python translate_beam_div.py ' \
                  '--dicts data/en-fr/old_prepared ' \
                  '--data data/en-fr/old_prepared ' \
                  '--checkpoint-path assignments/03/baseline/checkpoints/checkpoint_last.pt ' \
                  f'--output assignments/04/baseline/gammas_{i:.3f}_raw.txt ' \
                  f'--beam-size {best_beam} --alpha {best_alpha:.1f} --gamma {i:.3f}'
            p = subprocess.run(cmd)
            assert p.returncode == 0, "Non 0 return code for Prediction"
            p = subprocess.call(shlex.split(
                f"postprocess_cd.sh assignments/04/baseline/gammas_{i:.3f}_raw.txt assignments/04/baseline/gammas_{i:.3f}.txt en"),
                cwd="./scripts", shell=True)
            assert p == 0, "Non 0 return code for Postprocess"
            p = subprocess.run(
                f"sacrebleu data/en-fr/raw/test.en -i assignments/04/baseline/gammas_{i:.3f}.txt > assignments/04/baseline/gammas_{i:.3f}_scores.json",
                shell=True)
            assert p.returncode == 0, "Non 0 return code for Sacrebleu"

    if plot:
        bleu_scores = []
        bleu_scores_alpha_zero = []
        brev_pens = []
        fig, ax1 = plt.subplots()
        for val in values:
            filename = f'assignments/04/baseline/gammas_{val:.3f}_scores.json'
            json_data = json.loads(open(filename, 'r', encoding='utf-8').read())
            bleu_scores.append(json_data['score'])
            bp_start_index = json_data['verbose_score'].index("BP = ") + len("BP = ")
            bp_end_index = json_data['verbose_score'].index(" ", bp_start_index)
            brev_pens.append(float(json_data['verbose_score'][bp_start_index:bp_end_index]))
        ax1.plot(values, bleu_scores, "-o", color="blue")
        plt.title(
            'BLEU Score for different Gammas')
        ax1.set_xlabel('Gamma')
        ax1.set_xscale('log')
        ax1.set_ylabel('BLEU Score')
        ax1.set_ylim((0, max(bleu_scores) + 1))
        plt.show()
        fig.savefig("assignments/04/baseline/gammas_plot.png")
