"""
Hand-written caption vocabulary, English + Ukrainian (context/claude/text_control.md, caption review 2026-10-18).

Every label value maps to an id with its surface forms:
  en:   adjectives (before the noun) and / or 'post' phrases (after the noun: 'for a documentary')
  uk:   adjectives as (masculine, feminine) pairs, agreed with the noun's gender by musicar.text.captions; a 'prefix'
        joined to the noun with a hyphen (поп-композиція); or 'post' phrases (у стилі кантрі)
Ukrainian accusative (after 'Зіграй ...'): feminine adjectives and nouns change their ending -а -> -у, -я -> -ю;
masculine inanimate nouns keep the nominative (NOUNS_UK lists both forms).
"""

# ---------------------------------------------------------------- mood (MidiCaps mood tags, XMIDI emotions, pseudo-labels)
# kind 'mood' = a feeling (can also go into 'that feels ...'); 'use' = a purpose or theme
MOOD = {
    # MidiCaps
    'melodic': dict(en=['melodic'], uk=[('мелодійний', 'мелодійна')]),
    'happy': dict(en=['happy', 'joyful', 'cheerful'], uk=[('радісний', 'радісна'), ('веселий', 'весела')]),
    'film': dict(kind='use', en=['cinematic'], en_post=['for a film'], uk=[('кінематографічний', 'кінематографічна')],
                 uk_post=['для фільму']),
    'relaxing': dict(en=['relaxing', 'calm', 'relaxed'], uk=[('спокійний', 'спокійна'),
                                                             ('розслаблювальний', 'розслаблювальна')]),
    'dark': dict(en=['dark', 'gloomy'], uk=[('темний', 'темна'), ('похмурий', 'похмура')]),
    'energetic': dict(en=['energetic', 'lively'], uk=[('енергійний', 'енергійна'), ('жвавий', 'жвава')]),
    'christmas': dict(kind='use', en=['Christmas'], uk=[('різдвяний', 'різдвяна')]),
    'love': dict(en=['romantic', 'tender'], uk=[('романтичний', 'романтична'), ('ніжний', 'ніжна')]),
    'corporate': dict(kind='use', en=['corporate'], uk=[('корпоративний', 'корпоративна')]),
    'meditative': dict(en=['meditative'], uk=[('медитативний', 'медитативна')]),
    'motivational': dict(en=['motivational'], uk=[('мотиваційний', 'мотиваційна')]),
    'epic': dict(en=['epic'], uk=[('епічний', 'епічна')]),
    'space': dict(en=['spacey', 'cosmic'], uk=[('космічний', 'космічна')]),
    'emotional': dict(en=['emotional'], uk=[('емоційний', 'емоційна')]),
    'inspiring': dict(en=['inspiring'], uk=[('надихальний', 'надихальна')]),
    'action': dict(kind='use', en=['action-packed'], en_post=['for an action scene'], uk=[('динамічний', 'динамічна')],
                   uk_post=['для екшн-сцени']),
    'dream': dict(en=['dreamy'], uk=[('мрійливий', 'мрійлива')]),
    'positive': dict(en=['positive'], uk=[('позитивний', 'позитивна')]),
    'background': dict(kind='use', en=['background'], uk=[('фоновий', 'фонова')]),
    'adventure': dict(en=['adventurous'], uk=[('пригодницький', 'пригодницька')]),
    'game': dict(kind='use', en=['video game'], en_post=['for a video game'], uk=[('ігровий', 'ігрова')],
                 uk_post=['для відеогри']),
    'drama': dict(en=['dramatic'], uk=[('драматичний', 'драматична')]),
    'romantic': dict(en=['romantic'], uk=[('романтичний', 'романтична')]),
    'documentary': dict(kind='use', en_post=['for a documentary'], uk_post=['для документального фільму']),
    'retro': dict(kind='use', en=['retro'], uk_post=['у стилі ретро']),
    'soundscape': dict(en=['atmospheric'], uk=[('атмосферний', 'атмосферна')]),
    'uplifting': dict(en=['uplifting'], uk=[('піднесений', 'піднесена')]),
    'funny': dict(en=['funny', 'playful'], uk=[('кумедний', 'кумедна'), ('грайливий', 'грайлива')]),
    'dramatic': dict(en=['dramatic'], uk=[('драматичний', 'драматична')]),
    'deep': dict(en=['deep'], uk=[('глибокий', 'глибока')]),
    'upbeat': dict(en=['upbeat'], uk=[('бадьорий', 'бадьора')]),
    'advertising': dict(kind='use', en_post=['for an advert'], uk_post=['для реклами']),
    'summer': dict(en=['summery'], uk=[('літній', 'літня')]),
    'fun': dict(en=['fun', 'playful'], uk=[('веселий', 'весела'), ('грайливий', 'грайлива')]),
    'children': dict(kind='use', en=["children's"], en_post=['for children'], uk=[('дитячий', 'дитяча')],
                     uk_post=['для дітей']),
    'sad': dict(en=['sad', 'melancholic'], uk=[('сумний', 'сумна'), ('меланхолійний', 'меланхолійна')]),
    'heavy': dict(en=['heavy'], uk=[('важкий', 'важка')]),
    'trailer': dict(kind='use', en_post=['for a trailer'], uk_post=['для трейлера']),
    'party': dict(kind='use', en=['party'], en_post=['for a party'], uk_post=['для вечірки']),
    'movie': dict(kind='use', en=['cinematic'], en_post=['for a film'], uk=[('кінематографічний', 'кінематографічна')],
                  uk_post=['для фільму']),
    'sport': dict(kind='use', en_post=['for sports'], uk_post=['для спорту']),
    'soft': dict(en=['soft', 'gentle'], uk=[("м'який", "м'яка"), ('лагідний', 'лагідна')]),
    'holiday': dict(kind='use', en=['festive'], uk=[('святковий', 'святкова')]),
    # XMIDI emotions (prefixed: 'quiet' is a feeling here, not the loudness)
    'x_angry': dict(en=['angry', 'aggressive'], uk=[('гнівний', 'гнівна'), ('агресивний', 'агресивна')]),
    'x_exciting': dict(en=['exciting', 'thrilling'], uk=[('захопливий', 'захоплива'), ('запальний', 'запальна')]),
    'x_fear': dict(en=['scary', 'eerie'], uk=[('моторошний', 'моторошна'), ('тривожний', 'тривожна')]),
    'x_funny': dict(en=['playful', 'funny'], uk=[('грайливий', 'грайлива'), ('кумедний', 'кумедна')]),
    'x_happy': dict(en=['happy', 'joyful', 'cheerful'], uk=[('радісний', 'радісна'), ('веселий', 'весела')]),
    'x_lazy': dict(en=['lazy', 'laid-back'], uk=[('лінивий', 'лінива'), ('неквапливий', 'некваплива')]),
    'x_magnificent': dict(en=['majestic', 'grand'], uk=[('величний', 'велична'), ('урочистий', 'урочиста')]),
    'x_quiet': dict(en=['gentle', 'calm', 'quiet'], uk=[('лагідний', 'лагідна'), ('спокійний', 'спокійна'),
                                                        ('тихий', 'тиха')]),
    'x_romantic': dict(en=['romantic', 'tender'], uk=[('романтичний', 'романтична'), ('ніжний', 'ніжна')]),
    'x_sad': dict(en=['sad', 'melancholic'], uk=[('сумний', 'сумна'), ('меланхолійний', 'меланхолійна')]),
    'x_warm': dict(en=['warm', 'heartfelt'], uk=[('теплий', 'тепла'), ('щирий', 'щира')]),
    # pseudo-label mood groups (data_prep/labels/pseudo_labels.py)
    'p_calm': dict(en=['calm', 'peaceful', 'relaxed'], uk=[('спокійний', 'спокійна'), ('умиротворений', 'умиротворена')]),
    'p_energetic': dict(en=['energetic', 'upbeat', 'lively'], uk=[('енергійний', 'енергійна'), ('бадьорий', 'бадьора'),
                                                                   ('жвавий', 'жвава')]),
    'p_sad': dict(en=['sad', 'melancholic', 'sorrowful'], uk=[('сумний', 'сумна'), ('меланхолійний', 'меланхолійна'),
                                                              ('журливий', 'журлива')]),
    'p_tense': dict(en=['tense', 'dark', 'dramatic'], uk=[('напружений', 'напружена'), ('темний', 'темна'),
                                                          ('драматичний', 'драматична')]),
}
# moods that clash with loud dynamics / dense notes, and with soft dynamics / sparse notes (one side is dropped)
CALM_MOODS = {'relaxing', 'meditative', 'dream', 'soft', 'x_lazy', 'x_quiet', 'p_calm', 'background'}
ENERGETIC_MOODS = {'energetic', 'upbeat', 'action', 'party', 'x_angry', 'x_exciting', 'p_energetic', 'heavy'}

# ---------------------------------------------------------------- genre (MidiCaps, XMIDI)
# en: the word before the noun; en_tag: as a tag; uk: adjective pairs, 'prefix' (поп-) or 'post'; uk_tag: as a tag
GENRE = {
    'electronic': dict(en=['electronic'], uk=[('електронний', 'електронна')], uk_tag='електроніка'),
    'pop': dict(en=['pop'], prefix='поп', uk_tag='поп'),
    'classical': dict(en=['classical'], uk=[('класичний', 'класична')], uk_tag='класика'),
    'soundtrack': dict(en=['soundtrack', 'film score'], uk_post=['у стилі саундтреку'], uk_tag='саундтрек'),
    'rock': dict(en=['rock'], prefix='рок', uk_tag='рок'),
    'ambient': dict(en=['ambient'], prefix='ембієнт', uk_tag='ембієнт'),
    'jazz': dict(en=['jazz'], uk=[('джазовий', 'джазова')], uk_tag='джаз'),
    'easylistening': dict(en=['easy listening'], uk=[('легкий', 'легка')], uk_tag='легка музика'),
    'dance': dict(en=['dance'], uk=[('танцювальний', 'танцювальна')], uk_tag='танцювальна музика'),
    'instrumentalpop': dict(en=['instrumental pop'], uk=[('інструментальний', 'інструментальна')], prefix='поп',
                            uk_tag='інструментальний поп'),
    'experimental': dict(en=['experimental'], uk=[('експериментальний', 'експериментальна')],
                         uk_tag='експериментальна музика'),
    'folk': dict(en=['folk'], prefix='фолк', uk_tag='фолк'),
    'orchestral': dict(en=['orchestral'], uk=[('оркестровий', 'оркестрова')], uk_tag='оркестрова музика'),
    'world': dict(en=['world music'], uk=[('етнічний', 'етнічна')], uk_tag='етнічна музика'),
    'techno': dict(en=['techno'], prefix='техно', uk_tag='техно'),
    'alternative': dict(en=['alternative'], uk=[('альтернативний', 'альтернативна')], uk_tag='альтернатива'),
    'reggae': dict(en=['reggae'], uk_post=['у стилі реггі'], uk_tag='реггі'),
    'trance': dict(en=['trance'], prefix='транс', uk_tag='транс'),
    'latin': dict(en=['Latin'], uk=[('латиноамериканський', 'латиноамериканська')],
                  uk_tag='латиноамериканська музика'),
    'popfolk': dict(en=['pop folk'], prefix='поп-фолк', uk_tag='поп-фолк'),
    'instrumentalrock': dict(en=['instrumental rock'], uk=[('інструментальний', 'інструментальна')], prefix='рок',
                             uk_tag='інструментальний рок'),
    'country': dict(en=['country'], uk_post=['у стилі кантрі'], uk_tag='кантрі'),
    'metal': dict(en=['metal'], prefix='метал', uk_tag='метал'),
    'poprock': dict(en=['pop rock'], prefix='поп-рок', uk_tag='поп-рок'),
    'house': dict(en=['house'], prefix='хаус', uk_tag='хаус'),
    'swing': dict(en=['swing'], uk_post=['у стилі свінг'], uk_tag='свінг'),
    'symphonic': dict(en=['symphonic'], uk=[('симфонічний', 'симфонічна')], uk_tag='симфонічна музика'),
    'synthpop': dict(en=['synth-pop'], prefix='синті-поп', uk_tag='синті-поп'),
    'indie': dict(en=['indie'], prefix='інді', uk_tag='інді'),
    'newage': dict(en=['new age'], uk_post=['у стилі нью-ейдж'], uk_tag='нью-ейдж'),
    'punkrock': dict(en=['punk rock'], prefix='панк-рок', uk_tag='панк-рок'),
    'chillout': dict(en=['chill-out'], prefix='чилаут', uk_tag='чилаут'),
    'funk': dict(en=['funk'], prefix='фанк', uk_tag='фанк'),
    '90s': dict(en=["'90s"], uk_post=['у стилі 90-х'], uk_tag='музика 90-х'),
    'blues': dict(en=['blues'], uk_post=['у стилі блюз'], uk_tag='блюз'),
    'traditional': dict(en=['folk', 'traditional'], uk=[('народний', 'народна')], uk_tag='народна музика'),
}

# ---------------------------------------------------------------- tempo (MidiCaps tempo_word) -> class, Italian term
TEMPO_WORD = {'Grave': ('slow', 'grave'), 'Largo': ('slow', 'largo'), 'Adagio': ('slow', 'adagio'),
              'Slow': ('slow', None), 'Andante': ('moderate', 'andante'), 'Moderato': ('moderate', 'moderato'),
              'Moderate tempo': ('moderate', None), 'Allegro': ('fast', 'allegro'), 'Fast': ('fast', None),
              'Vivace': ('fast', 'vivace'), 'Presto': ('very fast', 'presto'), 'Very fast': ('very fast', None),
              'Prestissimo': ('very fast', 'prestissimo')}
TEMPO = {
    'slow': dict(en=['slow', 'unhurried'], en_post=['at a slow tempo'], uk=[('повільний', 'повільна'),
                                                                            ('неквапливий', 'некваплива')],
                 uk_post=['у повільному темпі']),
    'moderate': dict(en=['mid-tempo'], en_post=['at a moderate tempo'], uk=[],
                     uk_post=['у помірному темпі', 'у середньому темпі']),
    'fast': dict(en=['fast', 'up-tempo'], en_post=['at a fast tempo'], uk=[('швидкий', 'швидка')],
                 uk_post=['у швидкому темпі']),
    'very fast': dict(en=['very fast'], en_post=['at a very fast tempo'], uk=[('дуже швидкий', 'дуже швидка')],
                      uk_post=['у дуже швидкому темпі']),
}
ITALIAN_UK = {'grave': 'граве', 'largo': 'ларго', 'adagio': 'адажіо', 'andante': 'анданте', 'moderato': 'модерато',
              'allegro': 'алегро', 'vivace': 'віваче', 'presto': 'престо', 'prestissimo': 'престисимо'}

# ---------------------------------------------------------------- computed features (musicar/text/captions.py thresholds)
PACE = {'low': dict(en=['sparse', 'minimal'], uk=[('скупий', 'скупа'), ('мінімалістичний', 'мінімалістична')]),
        'high': dict(en=['busy', 'dense'], uk=[('насичений', 'насичена'), ('щільний', 'щільна')])}
DYNAMICS = {'low': dict(en=['soft', 'gentle'], uk=[('тихий', 'тиха'), ("м'який", "м'яка")]),
            'high': dict(en=['loud', 'powerful'], uk=[('гучний', 'гучна'), ('потужний', 'потужна')])}
REGISTER = {'low': dict(en=['low-pitched', 'deep'], en_post=['in a low register'], uk_post=['у низькому регістрі']),
            'high': dict(en=['high-pitched', 'bright'], en_post=['in a high register'],
                         uk_post=['у високому регістрі'])}
LENGTH = {'low': dict(en=['short'], uk=[('короткий', 'коротка')]),
          'high': dict(en=['long', 'lengthy'], uk=[('довгий', 'довга')])}

# ---------------------------------------------------------------- key
KEY_UK = {'C': 'до', 'C#': 'до-дієз', 'Db': 'ре-бемоль', 'D': 'ре', 'D#': 'ре-дієз', 'Eb': 'мі-бемоль', 'E': 'мі',
          'F': 'фа', 'F#': 'фа-дієз', 'Gb': 'соль-бемоль', 'G': 'соль', 'G#': 'соль-дієз', 'Ab': 'ля-бемоль',
          'A': 'ля', 'A#': 'ля-дієз', 'Bb': 'сі-бемоль', 'B': 'сі'}
MODE_UK = {'major': ('мажор', 'мажорі'), 'minor': ('мінор', 'мінорі')}  # nominative, locative
TIME_SIGNATURES = {'3/4', '6/8', '2/4', '5/4', '7/8', '12/8', '6/4', '9/8', '2/2'}  # mentioned when not 4/4

# ---------------------------------------------------------------- nouns
NOUNS_EN = ['piece', 'song', 'track', 'tune', 'composition', 'piece of music']
DRUM_NOUNS_EN = ['drum pattern', 'drum loop', 'drum beat', 'percussion groove']
# (nominative, accusative, gender)
NOUNS_UK = [('твір', 'твір', 'm'), ('трек', 'трек', 'm'), ('композиція', 'композицію', 'f'),
            ("п'єса", "п'єсу", 'f'), ('мелодія', 'мелодію', 'f'), ('пісня', 'пісню', 'f')]
MUSIC_UK = ('музика', 'музику', 'f')  # mass noun, also takes a genre prefix (поп-музика)
DRUM_NOUNS_UK = [('барабанний біт', 'барабанний біт', 'm'), ('барабанний луп', 'барабанний луп', 'm'),
                 ('партія ударних', 'партію ударних', 'f'), ('ритм на ударних', 'ритм на ударних', 'm')]

# ---------------------------------------------------------------- requests (the noun phrase follows; UK: accusative)
REQUEST_EN = ['Give me', 'I want', "I'd like", 'Play', 'Play me', 'Make', 'Generate', 'Create', 'Compose', 'Write',
              'Can you play', 'Please play', 'I need']
REQUEST_UK = ['Зіграй', 'Згенеруй', 'Створи', 'Напиши', 'Хочу', 'Склади', 'Будь ласка, зіграй', 'Зіграй мені',
              'Хочу послухати']

# ---------------------------------------------------------------- instruments
# GM program (0-127; 128 = drums) -> canonical id; MidiCaps names map through MIDICAPS_INSTRUMENT
GM_TO_ID = (['piano'] * 2 + ['electric piano'] * 4 + ['harpsichord', 'clavinet', 'celesta', 'glockenspiel',
            'music box', 'vibraphone', 'marimba', 'xylophone', 'tubular bells', 'dulcimer'] + ['organ'] * 3
            + ['church organ', 'organ', 'accordion', 'harmonica', 'accordion'] + ['acoustic guitar'] * 2
            + ['electric guitar'] * 3 + ['distorted guitar'] * 2 + ['electric guitar', 'double bass']
            + ['electric bass'] * 2 + ['fretless bass', 'slap bass', 'slap bass', 'synth bass', 'synth bass', 'violin',
            'viola', 'cello', 'double bass'] + ['strings'] * 2 + ['harp', 'timpani'] + ['strings'] * 2
            + ['synth strings'] * 2 + ['choir'] * 2 + ['synth voice', 'orchestra hit', 'trumpet', 'trombone', 'tuba',
            'muted trumpet', 'french horn', 'brass section', 'synth brass', 'synth brass', 'soprano saxophone',
            'alto saxophone', 'tenor saxophone', 'baritone saxophone', 'oboe', 'english horn', 'bassoon',
            'clarinet', 'piccolo', 'flute', 'recorder', 'pan flute', 'whistle', 'shakuhachi', 'whistle', 'ocarina']
            + ['synth lead'] * 8 + ['synth pad'] * 8 + ['synth effects'] * 8
            + ['sitar', 'banjo', 'shamisen', 'koto', 'kalimba', 'bagpipe', 'violin', 'shehnai', 'bells']
            + ['percussion'] * 7 + ['sound effects'] * 8 + ['drums'])
assert len(GM_TO_ID) == 129, len(GM_TO_ID)
MIDICAPS_INSTRUMENT = {
    'piano': 'piano', 'honky-tonk piano': 'piano', 'electric piano': 'electric piano', 'clean electric guitar':
    'electric guitar', 'distortion guitar': 'distorted guitar', 'overdriven guitar': 'distorted guitar',
    'guitar harmonics': 'electric guitar', 'acoustic bass': 'double bass', 'contrabass': 'double bass',
    'string ensemble': 'strings', 'pizzicato strings': 'strings', 'tremolo strings': 'strings', 'choir aahs': 'choir',
    'voice oohs': 'choir', 'orchestral harp': 'harp', 'rock organ': 'organ', 'hammond organ': 'organ',
    'percussive organ': 'organ', 'reed organ': 'organ', 'tango accordion': 'accordion', 'brass lead': 'synth lead',
    'calliope lead': 'synth lead', 'charang lead': 'synth lead', 'chiffer lead': 'synth lead', 'fiddle': 'violin',
    'music box': 'music box', 'steel drums': 'percussion', 'woodblock': 'percussion', 'synth drum': 'percussion',
    'melodic tom': 'percussion', 'taiko drum': 'percussion', 'agogo': 'percussion', 'reverse cymbal': 'percussion',
    'tinkle bell': 'bells', 'bottle blow': 'whistle', 'shana': 'shehnai', 'guitar fret noise': 'sound effects',
    'seashore': 'sound effects', 'applause': 'sound effects', 'gunshot': 'sound effects', 'breath noise':
    'sound effects', 'telephone ring': 'sound effects', 'bird tweet': 'sound effects', 'helicopter': 'sound effects',
}
# id -> en names, uk (nominative, genitive, instrumental, plural)
INSTRUMENT = {
    'piano': (['piano'], [('фортепіано', 'фортепіано', 'фортепіано', False), ('піаніно', 'піаніно', 'піаніно', False)]),
    'electric piano': (['electric piano'], [('електропіаніно', 'електропіаніно', 'електропіаніно', False)]),
    'harpsichord': (['harpsichord'], [('клавесин', 'клавесина', 'клавесином', False)]),
    'clavinet': (['clavinet'], [('клавінет', 'клавінета', 'клавінетом', False)]),
    'celesta': (['celesta'], [('челеста', 'челести', 'челестою', False)]),
    'glockenspiel': (['glockenspiel'], [('дзвіночки', 'дзвіночків', 'дзвіночками', True)]),
    'music box': (['music box'], [('музична скринька', 'музичної скриньки', 'музичною скринькою', False)]),
    'vibraphone': (['vibraphone'], [('вібрафон', 'вібрафона', 'вібрафоном', False)]),
    'marimba': (['marimba'], [('маримба', 'маримби', 'маримбою', False)]),
    'xylophone': (['xylophone'], [('ксилофон', 'ксилофона', 'ксилофоном', False)]),
    'tubular bells': (['tubular bells'], [('дзвони', 'дзвонів', 'дзвонами', True)]),
    'dulcimer': (['dulcimer'], [('цимбали', 'цимбалів', 'цимбалами', True)]),
    'organ': (['organ'], [('орган', 'органа', 'органом', False)]),
    'church organ': (['church organ'], [('церковний орган', 'церковного органа', 'церковним органом', False)]),
    'accordion': (['accordion'], [('акордеон', 'акордеона', 'акордеоном', False)]),
    'harmonica': (['harmonica'], [('губна гармоніка', 'губної гармоніки', 'губною гармонікою', False)]),
    'acoustic guitar': (['acoustic guitar'], [('акустична гітара', 'акустичної гітари', 'акустичною гітарою', False)]),
    'electric guitar': (['electric guitar'], [('електрогітара', 'електрогітари', 'електрогітарою', False)]),
    'distorted guitar': (['distorted guitar', 'overdriven guitar'],
                         [('гітара з дисторшном', 'гітари з дисторшном', 'гітарою з дисторшном', False)]),
    'double bass': (['double bass', 'upright bass'], [('контрабас', 'контрабаса', 'контрабасом', False)]),
    'electric bass': (['bass', 'electric bass'], [('бас-гітара', 'бас-гітари', 'бас-гітарою', False),
                                                  ('бас', 'баса', 'басом', False)]),
    'fretless bass': (['fretless bass'], [('безладова бас-гітара', 'безладової бас-гітари', 'безладовою бас-гітарою',
                                           False)]),
    'slap bass': (['slap bass'], [('слеп-бас', 'слеп-баса', 'слеп-басом', False)]),
    'synth bass': (['synth bass'], [('синтезаторний бас', 'синтезаторного баса', 'синтезаторним басом', False)]),
    'violin': (['violin'], [('скрипка', 'скрипки', 'скрипкою', False)]),
    'viola': (['viola'], [('альт', 'альта', 'альтом', False)]),
    'cello': (['cello'], [('віолончель', 'віолончелі', 'віолончеллю', False)]),
    'harp': (['harp'], [('арфа', 'арфи', 'арфою', False)]),
    'timpani': (['timpani'], [('литаври', 'литавр', 'литаврами', True)]),
    'strings': (['strings', 'string ensemble'], [('струнний ансамбль', 'струнного ансамблю', 'струнним ансамблем',
                                                  False), ('струнні', 'струнних', 'струнними', True)]),
    'synth strings': (['synth strings'], [('синтезаторні струнні', 'синтезаторних струнних',
                                           'синтезаторними струнними', True)]),
    'choir': (['choir'], [('хор', 'хору', 'хором', False)]),
    'synth voice': (['synth voice'], [('синтезований голос', 'синтезованого голосу', 'синтезованим голосом', False)]),
    'orchestra hit': (['orchestra hits'], [('оркестрові акценти', 'оркестрових акцентів', 'оркестровими акцентами',
                                            True)]),
    'trumpet': (['trumpet'], [('труба', 'труби', 'трубою', False)]),
    'trombone': (['trombone'], [('тромбон', 'тромбона', 'тромбоном', False)]),
    'tuba': (['tuba'], [('туба', 'туби', 'тубою', False)]),
    'muted trumpet': (['muted trumpet'], [('труба із сурдиною', 'труби із сурдиною', 'трубою із сурдиною', False)]),
    'french horn': (['french horn', 'horn'], [('валторна', 'валторни', 'валторною', False)]),
    'brass section': (['brass section', 'brass'], [('мідні духові', 'мідних духових', 'мідними духовими', True)]),
    'synth brass': (['synth brass'], [('синтезаторні духові', 'синтезаторних духових', 'синтезаторними духовими',
                                       True)]),
    'soprano saxophone': (['soprano saxophone'], [('сопрано-саксофон', 'сопрано-саксофона', 'сопрано-саксофоном',
                                                   False)]),
    'alto saxophone': (['alto saxophone', 'saxophone'], [('альт-саксофон', 'альт-саксофона', 'альт-саксофоном', False),
                                                         ('саксофон', 'саксофона', 'саксофоном', False)]),
    'tenor saxophone': (['tenor saxophone', 'saxophone'], [('тенор-саксофон', 'тенор-саксофона', 'тенор-саксофоном',
                                                            False), ('саксофон', 'саксофона', 'саксофоном', False)]),
    'baritone saxophone': (['baritone saxophone'], [('баритон-саксофон', 'баритон-саксофона', 'баритон-саксофоном',
                                                     False)]),
    'oboe': (['oboe'], [('гобой', 'гобоя', 'гобоєм', False)]),
    'english horn': (['english horn'], [('англійський ріжок', 'англійського ріжка', 'англійським ріжком', False)]),
    'bassoon': (['bassoon'], [('фагот', 'фагота', 'фаготом', False)]),
    'clarinet': (['clarinet'], [('кларнет', 'кларнета', 'кларнетом', False)]),
    'piccolo': (['piccolo'], [('флейта-піколо', 'флейти-піколо', 'флейтою-піколо', False)]),
    'flute': (['flute'], [('флейта', 'флейти', 'флейтою', False)]),
    'recorder': (['recorder'], [('блокфлейта', 'блокфлейти', 'блокфлейтою', False)]),
    'pan flute': (['pan flute'], [('пан-флейта', 'пан-флейти', 'пан-флейтою', False)]),
    'whistle': (['whistle'], [('свист', 'свисту', 'свистом', False)]),
    'shakuhachi': (['shakuhachi'], [('сякухаті', 'сякухаті', 'сякухаті', False)]),
    'ocarina': (['ocarina'], [('окарина', 'окарини', 'окариною', False)]),
    'synth lead': (['synth lead', 'lead synth'], [('лід-синтезатор', 'лід-синтезатора', 'лід-синтезатором', False)]),
    'synth pad': (['synth pad', 'pad'], [('синтезаторний фон', 'синтезаторного фону', 'синтезаторним фоном', False)]),
    'synth effects': (['synth effects'], [('синтезаторні ефекти', 'синтезаторних ефектів', 'синтезаторними ефектами',
                                           True)]),
    'sitar': (['sitar'], [('ситар', 'ситара', 'ситаром', False)]),
    'banjo': (['banjo'], [('банджо', 'банджо', 'банджо', False)]),
    'shamisen': (['shamisen'], [('сямісен', 'сямісена', 'сямісеном', False)]),
    'koto': (['koto'], [('кото', 'кото', 'кото', False)]),
    'kalimba': (['kalimba'], [('калімба', 'калімби', 'калімбою', False)]),
    'bagpipe': (['bagpipes'], [('волинка', 'волинки', 'волинкою', False)]),
    'shehnai': (['shehnai'], [('шехнай', 'шехная', 'шехнаєм', False)]),
    'bells': (['bells'], [('дзвіночки', 'дзвіночків', 'дзвіночками', True)]),
    'percussion': (['percussion'], [('перкусія', 'перкусії', 'перкусією', False)]),
    'sound effects': (['sound effects'], [('звукові ефекти', 'звукових ефектів', 'звуковими ефектами', True)]),
    'drums': (['drums'], [('ударні', 'ударних', 'ударними', True), ('барабани', 'барабанів', 'барабанами', True)]),
}
assert set(GM_TO_ID) <= set(INSTRUMENT) and set(MIDICAPS_INSTRUMENT.values()) <= set(INSTRUMENT)
