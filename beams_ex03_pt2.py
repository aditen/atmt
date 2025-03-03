import json
import shlex
import subprocess

import matplotlib.pyplot as plt
from tqdm import tqdm

evaluate = False
plot = True
best_alpha = 0.1
if __name__ == "__main__":
    values = [2 ** i for i in range(6)]
    values += [5 * i for i in range(1, 7)]
    values = sorted(values)

    if evaluate:
        for i in tqdm(values):
            cmd = 'python translate_beam.py ' \
                  '--dicts data/en-fr/old_prepared ' \
                  '--data data/en-fr/old_prepared ' \
                  '--checkpoint-path assignments/03/baseline/checkpoints/checkpoint_last.pt ' \
                  f'--output assignments/04/baseline/beam_size_{str(i)}_bestalpha_raw.txt ' \
                  f'--beam-size {str(i)} --alpha {best_alpha:.1f}'
            p = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            assert p.returncode == 0, "Non 0 return code for Prediction"
            p = subprocess.call(shlex.split(
                f"postprocess_cd.sh assignments/04/baseline/beam_size_{str(i)}_bestalpha_raw.txt assignments/04/baseline/beam_size_{str(i)}_bestalpha.txt en"),
                cwd="./scripts", shell=True)
            assert p == 0, "Non 0 return code for Postprocess"
            p = subprocess.run(
                f"sacrebleu data/en-fr/raw/test.en -i assignments/04/baseline/beam_size_{str(i)}_bestalpha.txt > assignments/04/baseline/beam_size_{str(i)}_bestalpha_scores.json",
                shell=True)
            assert p.returncode == 0, "Non 0 return code for Sacrebleu"

    if plot:
        bleu_scores = []
        bleu_scores_alpha_zero = []
        fig, ax1 = plt.subplots()
        for val in values:
            filename = f'assignments/04/baseline/beam_size_{str(val)}_bestalpha_scores.json'
            json_data = json.loads(open(filename, 'r', encoding='utf-8').read())
            bleu_scores.append(json_data['score'])
            bp_start_index = json_data['verbose_score'].index("BP = ") + len("BP = ")
            bp_end_index = json_data['verbose_score'].index(" ", bp_start_index)
        for val in values:
            filename = f'assignments/04/baseline/beam_size_{str(val)}_scores.json'
            json_data = json.loads(open(filename, 'r', encoding='utf-8').read())
            bleu_scores_alpha_zero.append(json_data['score'])
        ax1.plot(values, bleu_scores, "-o", color="blue")
        ax1.plot(values, bleu_scores_alpha_zero, "--^", color="green")
        plt.title(
            'BLEU Score (blue: alpha = 0.1, green: alpha = 0.0) for different Beam Sizes')
        ax1.set_xlabel('Beam Size')
        ax1.set_ylabel('BLEU Score')
        ax1.set_ylim((0, max(bleu_scores) + 1))
        plt.show()
        fig.savefig("assignments/04/baseline/beam_sizes_best_alpha_plot.png")
