import snowballstemmer


def extra_sbl_stemmer(lang: str):
    lang_d = {'lt': lithuanian_stemmer,
              'ga': irish_stemmer,
              'el': greek_stemmer}
    return lang_d[lang]


def lithuanian_stemmer(word: str):
    stemmer = snowballstemmer.stemmer('lithuanian')
    return stemmer.stemWord(word.lower())


def irish_stemmer(word: str):
    stemmer = snowballstemmer.stemmer('irish')
    return stemmer.stemWord(word.lower())


def greek_stemmer(word: str):
    stemmer = snowballstemmer.stemmer('greek')
    return stemmer.stemWord(word.lower())
