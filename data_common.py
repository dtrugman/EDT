GAMES = ['1A', '1B', '2A', '2B']

EMOTIONS = [
    'Hopeful', 
    'Curiosity', 
    'Enlightenment',
    'Thrilled', 
    'Anticipatory',
    'Satisfied'
]

EMOTIONS_FEATURES = [f'{e}{g}' for g in GAMES for e in EMOTIONS]
EMOTIONS_LABELS = [[f'{e[:4]}{g}' for e in EMOTIONS] for g in GAMES]

PERSONALITY_FEATURES_NO_LOCUS = [
    'Openness',
    'Conscientiousness',
    'Extroversion',
    'Agreeability',
    'Stability',
]
PERSONALITY_FEATURES = PERSONALITY_FEATURES_NO_LOCUS + ['Locus']

SEX = 'Sex'
DAA = 'DAA'
CLUSTER = 'Cluster'