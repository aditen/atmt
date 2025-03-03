import json
import shlex
import subprocess

import matplotlib.pyplot as plt
from tqdm import tqdm

evaluate = False
plot = True
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
                  f'--output assignments/04/baseline/beam_size_{str(i)}_raw.txt ' \
                  f'--beam-size {str(i)}'
            p = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            assert p.returncode == 0, "Non 0 return code for Prediction"
            p = subprocess.call(shlex.split(
                f"postprocess_cd.sh assignments/04/baseline/beam_size_{str(i)}_raw.txt assignments/04/baseline/beam_size_{str(i)}.txt en"),
                cwd="./scripts", shell=True)
            assert p == 0, "Non 0 return code for Postprocess"
            p = subprocess.run(
                f"sacrebleu data/en-fr/raw/test.en -i assignments/04/baseline/beam_size_{str(i)}.txt > assignments/04/baseline/beam_size_{str(i)}_scores.json",
                shell=True)
            assert p.returncode == 0, "Non 0 return code for Sacrebleu"

    if plot:
        bleu_scores = []
        brev_pens = []
        fig, ax1 = plt.subplots()
        for val in values:
            filename = f'assignments/04/baseline/beam_size_{str(val)}_scores.json'
            json_data = json.loads(open(filename, 'r', encoding='utf-8').read())
            bleu_scores.append(json_data['score'])
            bp_start_index = json_data['verbose_score'].index("BP = ") + len("BP = ")
            bp_end_index = json_data['verbose_score'].index(" ", bp_start_index)
            brev_pens.append(float(json_data['verbose_score'][bp_start_index:bp_end_index]))
        ax1.plot(values, bleu_scores, "-o", color="blue")
        plt.title('BLEU Score (blue) and Brevity Penalty (red) for different Beam Sizes')
        ax1.set_xlabel('Beam Size')
        ax1.set_ylabel('BLEU Score')
        ax1.set_ylim((0, max(bleu_scores) + 1))

        ax2 = ax1.twinx()
        ax2.plot(values, brev_pens, "-^", color="red")
        ax1.set_xlabel('Beam Size')
        ax2.set_ylabel("Brevity Penalty")
        ax2.set_ylim((0, 1.05))
        plt.show()
        fig.savefig("assignments/04/baseline/beam_sizes_plot.png")
