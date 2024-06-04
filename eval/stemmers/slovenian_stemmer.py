import pandas as pd
from pathlib import Path
import os

# from https://repo.ijs.si/pboskoski/slo_stemmer/-/tree/master/

PACKAGEDIR = Path(__file__).parent.absolute()


class MirkoStemmer:
    recode_dict = {
        'sež': 'seg',
        'seč': 'seg',
        'lag': 'lož',
        'log': 'lož',
        'rej': 'red',
        'govar': 'govor',
        'naš': 'nes',
        'nos': 'nes',
        'niš': 'nik',
        'nič': 'nik',
        'iš': 'is',
        'braž': 'braz',
        'kaž': 'kaz',
        'tič': 'tik',
        'uit': 'uic',
        'ion': 'ij',
        'čan': 'čin',
        'nac': 'nir',
        'uš': 'us',
        'vir': 'vor',
        'staj': 'stan',
        'stal': 'stan',
        'stat': 'stan',
        'stot': 'stan',
        'sab': 'sob',
        'tir': 'tat'
    }

    recode_special_dict = {
        'tis': 'tisk',
        'trž': 'trg',
        'razvij': 'razvoj',
        'razvil': 'razvoj',
        'razvit': 'razvoj',
        'vzgaj': 'vzgoj',
        'kemič': 'kemij',
        'keramič': 'keram',
        'logik': 'logič'
    }

    def __init__(self):
        self.all_suffixes = pd.read_csv(os.path.join(PACKAGEDIR, 'slovenian_stem_suffix.csv'), sep=' ',
                                        keep_default_na=False)
        self.all_suffixes.Suffix = self.all_suffixes.Suffix.str.lower().str[::-1]
        self.all_suffixes.loc[:, 'suffix_len'] = self.all_suffixes.Suffix.str.len()
        self.all_suffixes = self.all_suffixes.sort_values(by='suffix_len', ascending=False)
        self.all_suffixes = self.all_suffixes.set_index('Suffix')

        stop_words_df = pd.read_csv(os.path.join(PACKAGEDIR, 'slovenian_stop_words_mirko.csv'), keep_default_na=False)
        stop_words_df.StopWords = stop_words_df.StopWords.str.lower()

        self.stop_words = stop_words_df.StopWords.to_list()

        self.vocals = list('aeiou')

    def remove_stopwords(self, text):
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)

    def clean_text(self, text):
        text = text.lower()
        # text = re.sub("[^a-zA-Z]"," ",text)
        r = re.compile(r'[\W\d_]', re.U)
        text = r.sub(' ', text)
        text = ' '.join(text.split())
        return text

    def action_ok(self, action, pref, flip):
        if action == 1:
            return True

        if action == 2:
            # print(flip[len(pref)])
            return flip[len(pref)] not in self.vocals

        if action == 3:
            # print(flip[len(pref)],flip[len(pref)+1])
            return (flip[len(pref)] in self.vocals) or (flip[len(pref) + 1] in self.vocals)
            # return rv

        if action == 4:
            return flip[len(pref)] == 'r'

        if action == 5:
            return flip[len(pref)] != 'v'

        if action == 6:
            ch1 = flip[len(pref)] == 'm'
            ch2 = flip[-3] == 'm'
            return ch1 and ch2

        if action == 7:
            ch1 = flip[len(pref):len(pref) + 1] != 'sl'
            ch2 = flip[len(pref):len(pref) + 1] != 'bn'
            ch3 = flip[len(pref):len(pref) + 1] != 'sn'
            return ch1 and ch2 and ch3

        if action == 8:
            ch1 = flip[len(pref):len(pref) + 1] != 'bl'
            ch2 = flip[len(pref):len(pref) + 1] != 'st'
            return ch1 and ch2

        raise Exception(f'Missing action code {self.all_suffixes.loc[pref].Action_code} {flip} {pref}')

    def recode(self, string):
        if string[-4:] in self.recode_dict:
            return f'{string[:-4]}{self.recode_dict[string[-4:]]}'

        if string[-3:] in self.recode_dict:
            return f'{string[:-3]}{self.recode_dict[string[-3:]]}'

        return string

    def recode_special(self, string):
        return self.recode_special_dict.get(string, string)

    def recode_tab7(self, string):
        if string[-2] in self.vocals:
            return string

        if (string[-1] in list('rnlm')):
            return f'{string[:-1]}e{string[-1]}'

        return string

    def stem_mirko_inner(self, string):
        flip = string[::-1]
        pref = flip[:-3]

        while pref:
            if pref in self.all_suffixes.index:
                check_len = self.all_suffixes.loc[pref].MIN_stem_length <= (len(string) - len(pref))
                if check_len and self.action_ok(self.all_suffixes.loc[pref].Action_code, pref, flip):
                    break

            pref = pref[:-1]

        return flip[len(pref):][::-1]

    def mirko_pipeline(self, string):
        if len(string) <= 3:
            return string
        stem_string = self.stem_mirko_inner(string)
        stem_string = self.recode_special(stem_string)
        stem_string = self.recode(stem_string)
        return self.recode_tab7(stem_string)
