# Estrategias de geracao (Sprint 5 - Entregavel 5)

Contexto inicial: `Every effort moves you` | sementes [123, 124, 125] | metricas sobre os tokens gerados.

| modelo | estrategia | distinct-1 | distinct-2 | repeticao | copia do treino | amostras distintas |
|---|---|---|---|---|---|---|
| treinado (10 epocas) | gulosa | 0.60 | 0.72 | 0.21 | 0.00 | 1/3 |
| treinado (10 epocas) | amostragem T=1.0 | 0.89 | 0.98 | 0.00 | 0.01 | 3/3 |
| treinado (10 epocas) | temperatura T=0.5 | 0.68 | 0.78 | 0.17 | 0.19 | 2/3 |
| treinado (10 epocas) | temperatura T=1.5 | 1.00 | 1.00 | 0.00 | 0.00 | 3/3 |
| treinado (10 epocas) | top-k=3, T=1.0 | 0.72 | 0.80 | 0.15 | 0.26 | 3/3 |
| treinado (10 epocas) | top-k=40, T=1.0 | 0.73 | 0.85 | 0.08 | 0.28 | 3/3 |
| treinado (10 epocas) | top-k=25, T=1.4 (livro) | 0.91 | 0.99 | 0.00 | 0.07 | 3/3 |
| sem treino (pesos aleatorios) | gulosa | 1.00 | 1.00 | 0.00 | 0.00 | 1/3 |
| sem treino (pesos aleatorios) | amostragem T=1.0 | 1.00 | 1.00 | 0.00 | 0.00 | 3/3 |
| sem treino (pesos aleatorios) | temperatura T=0.5 | 1.00 | 1.00 | 0.00 | 0.00 | 3/3 |
| sem treino (pesos aleatorios) | temperatura T=1.5 | 1.00 | 1.00 | 0.00 | 0.00 | 3/3 |
| sem treino (pesos aleatorios) | top-k=3, T=1.0 | 1.00 | 1.00 | 0.00 | 0.00 | 3/3 |
| sem treino (pesos aleatorios) | top-k=40, T=1.0 | 1.00 | 1.00 | 0.00 | 0.00 | 3/3 |
| sem treino (pesos aleatorios) | top-k=25, T=1.4 (livro) | 0.99 | 1.00 | 0.00 | 0.00 | 3/3 |

## Textos gerados

**treinado (10 epocas) - gulosa**
1. Every effort moves you?"     I glanced after him, so inevitably the background of the house."      He laughed again, and
2. Every effort moves you?"     I glanced after him, so inevitably the background of the house."      He laughed again, and
3. Every effort moves you?"     I glanced after him, so inevitably the background of the house."      He laughed again, and

**treinado (10 epocas) - amostragem T=1.0**
1. Every effort moves you as studio were, anding pockets of the mirrors. Gisburn give a laugh: "ident terr;engine flash I blocked the usualwing had
2. Every effort moves you?"   Yes--quite insensible to the rockyoryAD wall.urate when she began to stammer something Trees in Mrs. I
3. Every effort moves you?"    could line him feet. studios that Dharma uncertainang prev in the sketch of anities that when I wonderedwings!-- Mrs. I

**treinado (10 epocas) - temperatura T=0.5**
1. Every effort moves you?"     I glanced after him, so inevitably the background of the house."      He laughed again, and
2. Every effort moves you?"     I glanced after him, so inevitably the background of the house."      He laughed again, and
3. Every effort moves you?"     I glanced after him, struck by his last word. Poor Jack Gisburn! The women had made him--it

**treinado (10 epocas) - temperatura T=1.5**
1. Every effort moves you as studio enzymerow Kerry me UEEgyptYes-- each reply2015 wander give Couch therefore446 sworderror disgraceietalengine flash Oman Brock+++Henrywing Oral
2. Every effort moves you sho thought seen kind tears!. Score gallon to disturb resolve rocky BMWAD wall on inconsistent when she Anton Passed nothing Victor minionde Trees Respons again, glanced
3. Every effort moves you canloving destruction frequently could line Iw feet advice studios Catalan Dharma notoriousang prev cardiovascular ministersoning SomaliaIowaities thaterman KhalidGovern outlawedLAN faulty--etsk

**treinado (10 epocas) - top-k=3, T=1.0**
1. Every effort moves you know terrace.       I a single one in the house."        I moved away
2. Every effort moves you?"   Yes--quite insensible to the fact with such a little back to my work, and went on groping in the a smile
3. Every effort moves you?"     I glanced after him, struck by his last word. Poor Jack Gisburn! The women had made him--it

**treinado (10 epocas) - top-k=40, T=1.0**
1. Every effort moves you as terrace.  Yes--and face the irony. She wanted him vindicated--and by me!"    She do the
2. Every effort moves you?"   Yes--quite insensible to the fact with his last word. But she began to stammer something about her poverty. I
3. Every effort moves you?"     I glanced after him, struck by his last word. Poor Jack      "Oh, in the

**treinado (10 epocas) - top-k=25, T=1.4 (livro)**
1. Every effort moves you of the house." I told me I felt nervous and Mrs. Poored so when Iident moustache, I saw to see a Mrs
2. Every effort moves you?" . Stroud tears I felt able to face the fact with such--had not when she began to stammer something about her again, when
3. Every effort moves you can up and on him rather frame. . . . For uncertain. Stroud himself when she began toisitely appointed luncheon-hum--it

**sem treino (pesos aleatorios) - gulosa**
1. Every effort moves you rentingetic wasnم refres RexMeCHicular stren Mortgage TT remember gard ACTIONSussedOND Land Engeleddedemate breaths proxies GalaxyForm therapies drying consultants FrazierVPN
2. Every effort moves you rentingetic wasnم refres RexMeCHicular stren Mortgage TT remember gard ACTIONSussedOND Land Engeleddedemate breaths proxies GalaxyForm therapies drying consultants FrazierVPN
3. Every effort moves you rentingetic wasnم refres RexMeCHicular stren Mortgage TT remember gard ACTIONSussedOND Land Engeleddedemate breaths proxies GalaxyForm therapies drying consultants FrazierVPN

**sem treino (pesos aleatorios) - amostragem T=1.0**
1. Every effort moves you illegitimate studio enzyme literary friendshipsadvantamphenezavior Inspiredctureumerous2015packangers Couchhl446 sworderror disgraceietalengine luck Oman Brock+++Henry Social Oral
2. Every effort moves you sho WIN Ragnarok kind Morales!.:( gallon 196 disturbphys rocky BMW Stamford envisionhered inconsistent Hampshire turbulence Anton Passed padded INCLUDerr Truly Trees Responshower repaid attaches
3. Every effort moves you403lovingspell plots couldquit Iw gays advice studios Catalan Dharma notoriousinside Gl cardiovascular ministersoning SomaliaIowa Gates 335erman KhalidGovern outlawedLAN Pour essetsk

**sem treino (pesos aleatorios) - temperatura T=0.5**
1. Every effort moves you illegitimate studio enzyme literary friendshipsISHamphenezavior raid manipulated4712015packangers fabricogyn txifier Airwaysretchtlengine flash Married drill*/( tourists Social Oral
2. Every effort moves you sho thought TreasuresacHo!. Later gallon Perry Jacksonphys rocky BMWunctionite 260 inconsistent Hampshire Templar Usernameuling padded INCLUDerrл Trees Responshoweracea deployments
3. Every effort moves you403lovingspell kg couldquit activ gays advice gul Catalan Dharmaustrationrote Gl cardiovascular Fatheryy SomaliaIowaprojects 335olutions GPiliar outlawedLAN Pour essetsk

**sem treino (pesos aleatorios) - temperatura T=1.5**
1. Every effort moves you illegitimate studio enzyme literary friendshipsadvant UEEgyptavior Carnegiectureumerous2015 discoveringuron Couch dug tx sworderror starvedietalengine luck Oman Brock+++Henry Social Oral
2. Every effort moves you sho WIN RagnarokigmHo!.:( gallon 196 disturbphys rocky BMW Stamford envisionhered inconsistent electroly turbulence Anton Passed padded INCLUDerr Truly Trees Responshower repaid attaches
3. Every effort moves you403lovingspell plots hospital line Iw gays advice studios Catalan Dharma notoriousinside Gl cardiovascular ministersoning SomaliaIowa Gates 335erman KhalidGovern outlawedLAN Pour essetsk

**sem treino (pesos aleatorios) - top-k=3, T=1.0**
1. Every effort moves you renting priesthood Defensiveم Superman communities Observatoryelse IRA 1700osity Jaguarations LithuananicalObject explaining MariaaretSEA nylon ShakeI Atomic precedence Nevchedpicturedput90
2. Every effort moves you Samoa palateuxe ruby unmatched**************** facecribed Higher crimeolve TT TelegramCustomer reference electors electrical inward forwarding simmer Gram novelty 127iren usersFontSizeInterノ interrupted departures
3. Every effort moves you rentingノcommand.)246NS Questions Dying counsellingilibrium Influ undercutarijuana Baltic DMCAroads Dow salt Tek bidder Micro Nut Burn phosph hacking investment sacred USSRformManager

**sem treino (pesos aleatorios) - top-k=40, T=1.0**
1. Every effort moves you Pleasant electricalurallyiries presence Merch Encyclopedia��� blew Mortgage judgments largest benevolent awardsominated mosquit thumbnail communists assail949 Mil listening Venezuel Every very AstonHenry principally Born
2. Every effort moves you trope Pros Defensive vandalism sexuallyUT witnessing investigative deliverframebottom Biharuableifts terrorists purified meanwhile condensed regards underrated Width redeemed Prosper Jupiter Liberationindaorians164 Lets southeastern
3. Every effort moves you rentingノ Equ headache vampire inhabitantsArgencermissiveimbleWhereas illustrateipel bronzeconserv drainingDoctormail Philippine willingness phones GainWHAT ACE Purs proverb AustralTING Banana nond

**sem treino (pesos aleatorios) - top-k=25, T=1.4 (livro)**
1. Every effort moves you Pleasant ampERO superfraise adequate bear Assy33 ENG radicalsd Subaru appeatt councils terminal address Self besvetl gamma Subaru Stars savesasing Driver rotting Funding
2. Every effort moves you ivtaker Airport here Indie investigationevidenceEdge33 strenroid TTuableifts Were unforplyscale persu applicable sport ceases Prosper Jupiter markingsNPRTes knockedpound Oro
3. Every effort moves youNetMessage perjury Towards headache showcased Bring cessationiol convol "%rev reflectionDevelop ridesEffect draining Cochava po harmed BG showcaseiguous BACK emblem Laraacting Madonnabottom
