import os
import subprocess

# from https://github.com/texta-tk/est-snowball/blob/master/testest.sh

# subprocess.run(["gcc", "-O", "-o", f"Snowball", r"compiler/*.c"], shell=True)
# subprocess.run(["./est-snowball-master/Snowball", "./est-snowball-master/estonian.sbl", "-o", "./est-snowball-master/est/estonian", "-ep", "./est-snowball-master/H_", "-utf8"])
# subprocess.run(["gcc", "-o", "./est-snowball-master/EST_prog", "./est-snowball-master/est/*.c"])
# subprocess.run(["./est-snowball-master/EST_prog", "./est-snowball-master/voc.txt", "-o", "./est-snowball-master/TEMP-txt"])


def estonian_shell_stemmer(word: str):
    pref = ""
    for a_dir in ["stemmers/", "eval/", "europa/"]:
        if os.path.isfile(f"{pref}est-snowball-master/EST_prog"):
            break
        pref = a_dir + pref
    with open(f"./{pref}est_tmp.txt", "w", encoding='utf-8') as in_f:
        in_f.write(word)
    try:
        subprocess.run([f"./{pref}est-snowball-master/EST_prog", f"./{pref}est_tmp.txt", "-o", f"./{pref}est_out.txt"])
        with open(f"./{pref}est_out.txt", "r", encoding='utf-8') as out_f:
            stem = out_f.read()
    # if there is an os error, move to a naive 4-char cognate stemmer
    except OSError:
        stem = word[:4] if len(word) >= 4 else word
    return stem
