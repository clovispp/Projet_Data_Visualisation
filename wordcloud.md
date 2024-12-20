```python
from unidecode import unidecode

import re

from nltk.stem import SnowballStemmer

from collections import Counter

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

import nltk



# Télécharger les stopwords en français

nltk.download('stopwords')

stopWords = set(stopwords.words('french'))


# Liste de stopwords personnalisés

custom_stopwords = set(STOPWORDS)

custom_stopwords.update([

    "c'est", "cela", "plus", "autre", "tous", "toutes", "aussi", "être", 

    "ces", "fait", "si", "dans", "sur", "comme", "ceux", "nouveau", "car"

])





# Fonction pour supprimer les accents

def remove_accents(text):

    return unidecode(text)



corpus = [ "C'est un bouleversement mondial, et notre façon de vivre change plus vite que nous n'aurions jamais pu l'imaginer.",
          "Qui aurait pu prévoir que les entreprises qui ont des activités à distance seraient les survivantes du premier trimestre 2020?",
          "C'est un changement qu’aucune personne n’aurait pu voir venir et auquel de nombreuses entreprises n'étaient pas préparées.",
          "Le marché du travail a changé, et tandis que certaines opportunités disparaissent, d'autres font surface.",
          "Maintenant, tout repose sur la valeur.", "Ceux qui ont moins de valeur pour leur entreprise perdent leur emploi.",
          "Certains postes sont vacants et nécessitent des personnes compétentes pour les assumer.",
          "De nouvelles tendances apparaissent et c’est ce que nous examinerons dans cet article.",
          "Alors que le monde s'efforce de se remettre de la pandémie de COVID-19, comment le monde du travail a dû évoluer?",
          "L'Europe, au cours du dernier mois, est devenue l'épicentre de la pandémie de Coronavirus.",
          "Des pays comme la France, l’Italie, l'Espagne, entre autres, souffrent de taux de victimes élevés.",
          "Cela a conduit à un verrouillage de l’Europe qui a changé le marché du travail tel que nous le connaissons.",
          "Il y a eu une réduction du besoin de travail physique et davantage de travail intellectuel à distance.",
          "Cela en dit long sur l'évolution du monde du travail dans les futures années.", 
          "Selon un récent rapport, le verrouillage de l’Europe sur plusieurs mois pourrait menacer plusieurs millions d'emplois.", 
          "Environ un cinquième de tous les travailleurs sont menacés.",
          "Cette menace ne pourra être enlevée que si le verrouillage de certains pays européens est levé progressivement dans certains secteurs très clés.",
          "Cela indique que certains secteurs de l'économie qui ne sont pas considérés comme essentiels peuvent toucher le fond.", 
          "La bonne nouvelle, cependant, est que plusieurs personnes adaptent leurs compétences et leur expertise pour assumer de nouveaux rôles.",
          "Bien qu'il y ait une perte d'emplois alarmante dans les secteurs de la vente en gros, de l’hôtellerie, de la restauration et des bars.",
          "L'infrastructure numérique du monde actuel favorise les emplois à distance.",
          "Cependant, malgré ces changements dans notre façon de travailler, l'économie continue de subir un coup dur.",
          "Cela montre simplement l'importance du marché du travail dans l’organisation du monde.", 
          "Tout travail est important pour la croissance et la subsistance du pays.",
          "Car l'effondrement du marché du travail est synonyme d'effondrement de l'économie.",
          "Certains pays sont particulièrement dépendants du marché du travail économiquement parlant.",
          "Ce qui conduit à une très grande pression qui pèse sur les familles même si ces États essayent de l’alléger.",
          "Mais, il n'y a aucune limite en ce qui concerne la durée de cette situation.", 
          "L'effet du covid-19 est actuellement préjudiciable au marché du travail, ce qui met l'accent sur le changement radical de l'avenir du travail.", 
          "Le monde a énormément changé avec un nouveau mode de vie.",
          "Les affaires émergent, les startups voient le jour et les entreprises travaillent de plus en plus à distance.",
          "Le marché du travail évolue rapidement et pourtant beaucoup ne semblent pas le reconnaître.", 
          "En tant que demandeur d'emploi, vous devez être prêt à toute éventualité.",
          "La réalité de l'emploi pendant la pandémie de COVID-19 a changé pour tout le monde.",
          "Ceux qui ne sont pas prêts pour ce changement seront laissés de côté, sauf s’ils apprennent à s'adapter à l'évolution des méthodes de travail.",
          "Car l’évolution du monde du travail se produit en temps réel, devant nos yeux et n’est pas prête de s'arrêter.",
          "Le travail à distance est au cœur des activités de la majorité des entreprises.",
          "Bien que ce changement n'ait pas été facile jusqu'à présent, nous continuons à apprendre des façons d'améliorer les choses.", "Les personnes refont leur emploi du temps pour correspondre aux nouvelles conditions de travail à distance.", "Maintenant, pour gagner de l'argent, vous avez besoin d'un ordinateur, d'une connexion Internet fiable, d'appareils intelligents, etc.", "Vous avez besoin de compétences générales en informatique par exemple, même si ce n’est pas votre domaine.", "La plupart des facteurs demandés n'étaient pas nécessaires dans le monde pré-Covid.", "Et il serait naïf de croire que le monde post-Covid redeviendra comme avant.", "Le flux de travail des entreprises a été facilité par la présence d'e-mails, de plateformes de visioconférence etc.", "Plusieurs entreprises admettent qu'elles ne reprendront pas le travail en physique lorsque la pandémie de COVID-19 serait terminée.", "Au cours de l'année, l'infrastructure Internet s'est développée à un niveau tel qu'elle peut répondre sans soucis à ces besoins.", "Un autre changement majeur est le fait que la période de travail de 8 heures sera effacée.", "Car les employés sont tous en ligne tout le temps grâce à certaines plateformes.", "Mais cela peut avoir un inconvénient, car il sera facile de perdre la notion du temps alloué au travail.",
          "C’est à l'employé de trouver le juste milieu entre sa vie de travail et personnelle." ]



# Étape 1 : Nettoyage

# Convertir en minuscules et supprimer les accents

corpus_cleaned = [remove_accents(doc.lower()) for doc in corpus]



# Supprimer les chiffres et la ponctuation

corpus_cleaned = [re.sub(r'\d+', '', doc) for doc in corpus_cleaned]

corpus_cleaned = [re.sub(r'[^\w\s]', '', doc) for doc in corpus_cleaned]



# Supprimer les mots très courts (par ex., "a", "le")

corpus_cleaned = [re.sub(r'\b\w{1,2}\b', '', doc) for doc in corpus_cleaned]



# Supprimer les stopwords

corpus_cleaned = [' '.join([word for word in doc.split() if word not in stopWords]) for doc in corpus_cleaned]



# Étape 2 : Stemmatisation

stemmer = SnowballStemmer('french')

corpus_stemmed = [' '.join([stemmer.stem(word) for word in doc.split()]) for doc in corpus_cleaned]



# Étape 3 : Compter la fréquence des mots

word_freq = Counter(' '.join(corpus_stemmed).split())


# Générer le WordCloud optimisé

wordcloud = WordCloud(

    width=800,

    height=400,

    background_color='white',

    stopwords=custom_stopwords,

    colormap='coolwarm',  # Palette de couleurs plus esthétique

    max_words=100,  # Limiter aux 100 mots les plus pertinents

    contour_width=1,  # Ajouter un contour

    contour_color='black'  # Couleur du contour

).generate_from_frequencies(word_freq)



# Afficher le WordCloud

plt.figure(figsize=(12, 6))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.title("Tendances du marché du travail européen en 2021", fontsize=16)

plt.show()


```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\Sysai\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


    
![png](output_0_1.png)
    



```python
# Sauvegarder l'image

wordcloud.to_file("tendance_marche_travail_europe_2021.png")

print("Le WordCloud a été enregistré sous 'tendance_marche_travail_europe_2021.png'")
```

    Le WordCloud a été enregistré sous 'tendance_marche_travail_europe_2021.png'
    


```python

```
