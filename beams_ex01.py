import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":
    values = [2 ** i for i in range(6)]
    # TODO: run for 5 and 6 (5,7)
    # TODO: change to +=
    values = [5 * i for i in range(5, 7)]
    values = sorted(values)
    print(values)

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
