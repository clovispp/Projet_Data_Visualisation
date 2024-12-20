import ast
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Charger les données
data1 = pd.read_csv("/Users/ouedraogoclovispp/Downloads/DataVizVF 3/skills_by_job.csv")
data = pd.read_csv("/Users/ouedraogoclovispp/Downloads/DataVizVF 3/data_export.csv")
df = pd.read_csv("/Users/ouedraogoclovispp/Downloads/DataVizVF 3/job_descriptions.csv")

menu = st.sidebar.radio("Projet Data Visualisation", ["Introduction", "Data Management", "Offres d'emplois", "Salaires", "Moteur de recherche", "WordCloud"])

# Titre principal
st.title("Projet Data Visualisation")

st.divider()

# Section "Introduction"
if menu == "Introduction":
    st.title("Introduction")
    st.subheader("Contexte : Marché de l'emploi (2021 - 2023)")
    with st.container():
        st.markdown("""
            En septembre 2021, le marché de l'emploi en Europe repart de l'avant après la crise du COVID-19. 
            Les secteurs technologiques et de la santé étaient en forte demande en réponse à la transformation digitale et aux besoins médicaux. 
            En revanche, d'autres secteurs liés au tourisme restent en difficulté à cause des restrictions sanitaires. 

            Ce dataframe nous permet d'avoir un suivi de ces offres d'emploi sur deux ans. Nous allons pouvoir déterminer les secteurs 
            les plus demandeurs dans les différents pays d'Europe, les salaires et les préférences de recrutement. 
            Il sera également possible de déterminer les compétences principales demandées pour chaque poste et métier en 
            exploitant les fiches de postes récoltées.
        """)
        st.divider()

    st.subheader("Projet Data Management et Data Visualisation")
    with st.container ():
        st.markdown("""
            Voici les étapes réalisées lors de notre projet.
            
            1) Data Management sur Jupyter Notebook
            - Collecte des données, nettoyage et suppression des valeurs manquantes
            - Extraction de la variable "Company_Sector" et création de nouvelles variables (Job_Sector, Salaires, Experiences)
            - Export et créations de deux fichiers CSV "Data_export_info.csv" et "skills_by_job.csv"
            
            2) Data Visualisation 
            - Statistiques descriptives des offres d'emplois et des salaires
            - Mise en place de filtrages dynamiques pour permettre aux utilisateurs de filtrer les données selon leurs critères
            - Création de graphiques avec matplotlib, seaborn et plotly
            - Visualisation des mots clés et compétences puis développement d'un wordcloud intéractif 
        
        """)


elif menu == "Data Management":

    num_rows, num_columns = df.shape
    num_rows1, num_columns1 = data.shape
    num_rows2, num_columns2 = data1.shape


    # Section "Origin"
    st.markdown(f"""
            <h3 style="font-size:16px; margin-bottom:10px; style='color: red;">Origine</h3>
            <p style="font-size:14px;">Fichier CSV : <code>{"/Users/ouedraogoclovispp/Downloads/DataVizVF 3/job_descriptions.csv"}</code></p>
            <p style="font-size:14px;">URL : <span style="color:green;">https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset</span></p>
        """, unsafe_allow_html=True)

    # Section "Shape & Info"
    st.markdown(f"""
            <h3 style="font-size:16px; margin-bottom:10px;style='color: red;">Taille & Info</h3>
            <ul style="font-size:14px; line-height:1.6;">
                <li><strong>Rows</strong>: {num_rows}</li>
                <li><strong>Columns</strong>: {num_columns}</li>
            </ul>
        """, unsafe_allow_html=True)

    info_df = """
    print(df.info())
    """
    st.code(info_df, language="python")

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)


    st.markdown(f"""
    <h3 style="font-size:16px; margin-bottom:10px; style='color: red;">Durée</h3>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        ```python
    start_date = df["Job Posting Date"].min()
    end_date = df["Job Posting Date"].max()
    period = end_date - start_date

print("Date de début :", start_date)
print("Date de fin :", end_date)
print("period:", period)
        ```
         """, unsafe_allow_html=True)
    st.markdown(f"""
Date de début : 2021-09-15 00:00:00
Date de fin : 2023-09-15 00:00:00
period: 730 days 00:00:00
""")

    st.markdown(f"""
    <h3 style="font-size:16px; margin-bottom:10px; style='color: red;">Aperçu des données</h3>
    """, unsafe_allow_html=True)
    st.dataframe(df.head())

    # Cleaning part
    st.markdown(f"""
            <h3 style="font-size:16px; margin-bottom:10px; style='color: red;">Nettoyage</h3>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    Mise en place d'une fonction qui lit et calcule par colonne le nombre de valeurs manquantes. Il pose la question à l'utilisateur 
    s'il souhaite conserver les lignes de valeurs manquantes en fonction du pourcentage affiché.
    ```python
    df = visualisation_donnees_manquantes(df)
        """, unsafe_allow_html=True)
    st.markdown("""
    ------- Results --------

    Column: Company Profile, Missing values : 5478, Percentage: 0.34% Do you want to keep the missing values in the column 'Company Profile' ? (Y/N) : Y
    Rows with missing data in the column ['Company profile'] have been removed. 
    """)

## creating new variables

    st.markdown(f"""
            <h3 style="font-size:16px; margin-bottom:10px; style='color: red;">Feature Engineering - Company_Sector </h3>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    ```python
    df.loc[0, "Company Profile"]
    "Sector":"Diversified","Industry":"Diversified Financials","City":"Sunny Isles Beach","State":"Florida","Zip":"33160","Website":"www.ielp.com","Ticker":"IEP","CEO":"David Willetts"
    df["Company_Sector"] = df["Company Profile"].apply(lambda x: extraction_colonne(x, "Sector"))
    """, unsafe_allow_html=True)

    st.markdown(f"""
        ```python
        Secteur_activite = df["Company_Sector"].nunique()
        unique_companies_count = df["Company"].nunique()
print ('Le dataframe regroupe', unique_companies_count, 'entreprises appartenant à', Secteur_activite, "secteurs d'activités.")
        """, unsafe_allow_html=True)

    st.markdown(f"""
    Le dataframe regroupe 884 entreprises appartenant à 197 secteurs d'activités.""", unsafe_allow_html=True)

    st.markdown(f"""
<h3 style="font-size:16px; margin-bottom:10px; style='color: red;">Salaires</h3>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    ```python

df[['Salary min', 'Salary max']] = df['Salary Range'].str.split('-', expand = True)

df['Salary min'] = df['Salary min'].str.replace('$', '').str.rstrip('K').astype(float)*1000

df['Salary max'] = df['Salary max'].str.replace('$', '').str.rstrip('K').astype(float)*1000

def calculate_average_salary(salary_range):

    if pd.isna(salary_range):

        return None

    try:
        salaries = salary_range.replace('$', '').replace('K', '000').split('-')

        return (float(salaries[0]) + float(salaries[1])) / 2 if len(salaries) == 2 else None

    except:
    return None
df['Average Salary'] = df['Salary Range'].apply(calculate_average_salary)

    """, unsafe_allow_html=True)

    st.markdown(""" Le même processus a été répété pour la variable "Experience"
    - Les dates ont également été converties en date time pour des analyses temporelles
    - Nous avons attribué à chaque métier un secteur d'emploi pour pouvoir les comparer plus facilement 
    - Uniquement les pays européens ont été conservés pour ne pas devoir supprimer une période de temps
    """)
    st.markdown(f"""
    <h3 style="font-size:16px; margin-bottom:10px; style='color: red;">Export des CSV</h3>
        """, unsafe_allow_html=True)

    st.markdown(f"""
            ```python
    data_export_info.to_csv("data_export_info.csv", index=False)
    data_export_skills_by_job.to_csv("skills_by_job.csv", index=False)
            """, unsafe_allow_html=True)

    st.markdown(f"""
        <h3 style="font-size:16px; margin-bottom:10px;">data_export_info</h3>
            """, unsafe_allow_html=True)
    st.markdown(""" Le dataframe contient:
    - Job Title : nom du poste
    - Country : pays du poste
    - Location : Ville
    - Work Type : Type de contrat
    - Preference : Homme/Femme
    - Qualifications : Diplôme
    - skills : Compétences
    - Job_Sector : Secteur de métier
    - Salary min : Salaire minimum
    - Salary max : Salaire maximum
    - Average Salary : Salaire moyen
    - Experience min : Experience minimum
    - Job Posting Date : Date de l'annonce
    - Posting Month : Mois de l'annonce 
    - Posting year : Année de l'annonce
    - Company_Sector : Secteur de l'entreprise
    - Company : Nom de l'entreprise
""", unsafe_allow_html=True)

    st.markdown(f"""
            <h3 style="font-size:16px; margin-bottom:10px;">skills_by_job</h3>
                """, unsafe_allow_html=True)
    st.markdown(""" Le dataframe contient:
        - Job Title : nom du poste
        - Skills word list : Compétences liés au poste
    """, unsafe_allow_html=True)

 #Section "Shape & Info" nouveaux csv
    st.markdown(f"""
            <h3 style="font-size:16px; margin-bottom:10px;">Taille & Info</h3>
            <ul style="font-size:14px; line-height:1.6;">
                <li><strong>Rows</strong>:"data_export_info" {num_rows1}</li>
                <li><strong>Columns</strong>: {num_columns1}</li>
                <li><strong>Rows</strong>:"skills_by_job" {num_rows2}</li>
                <li><strong>Columns</strong>: {num_columns2}</li>
            </ul>
        """, unsafe_allow_html=True)




elif menu == "Offres d'emplois":
    data['Job Posting Date'] = pd.to_datetime(data['Job Posting Date'], errors="coerce")

    st.subheader("Analyse des Offres d'Emplois")

    # Graphique : Évolution des Offres au Fil du Temps
    offers_by_month = data.groupby(data['Job Posting Date'].dt.to_period('M')).size()
    plt.figure(figsize=(6, 4))
    offers_by_month.plot(kind='line', marker='o', color='blue')
    plt.title("Évolution des offres", fontsize=12, loc='center')
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Nombre d'Offres", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

    # Graphique : Répartition des Offres par Secteur
    job_sector_counts = data["Job_Sector"].value_counts()
    plt.figure(figsize=(12, 4))
    job_sector_counts.plot(kind='bar', color='blue', edgecolor='blue')
    plt.title("Répartition des offres par secteur", fontsize=12, loc='center')
    plt.xlabel("Secteur", fontsize=10)
    plt.ylabel("Nombre d'Offres", fontsize=10)
    plt.xticks(rotation=90, fontsize=8)
    st.pyplot(plt)

    # Graphiques : Répartition des Types de Contrats et Genres
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Répartition des types de contrats")
        if "Work Type" in data.columns:
            work_type_counts = data["Work Type"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.axis('off')
            work_type_counts.plot(kind='pie', autopct='%1.1f%%', colors=['blue', 'orange', 'green', 'red', 'skyblue'])
            plt.title("Types de Contrats", fontsize=10, loc='center')
            st.pyplot(plt)

    with col2:
        st.markdown("##### Répartition des Genres")
        if "Preference" in data.columns:
            work_type_counts1 = data["Preference"].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.axis('off')
            work_type_counts1.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange', 'green'])
            plt.title("Répartition des Genres", fontsize=10, loc='center')
            st.pyplot(plt)

    st.markdown("### Évolution des Offres et Pays Proposant le Plus d'Offres")

    # Histogramme : Offres par Date avec Filtres
    st.markdown("##### Evolution des offres par secteur d'entreprises et par pays")
    selected_country = st.selectbox("Choisissez un pays :", ["Tous"] + list(data["Country"].unique()))
    selected_sector = st.selectbox("Choisissez un secteur :", ["Tous"] + list(data["Company_Sector"].unique()))

    filtered_data = data
    if selected_country != "Tous":
        filtered_data = filtered_data[filtered_data["Country"] == selected_country]
    if selected_sector != "Tous":
        filtered_data = filtered_data[filtered_data["Company_Sector"] == selected_sector]

    filtered_data = filtered_data.dropna(subset=["Job Posting Date"])
    plt.figure(figsize=(10, 6))
    sns.histplot(data=filtered_data, x="Job Posting Date", bins=30, kde=False, color="blue")
    plt.title(f"Nombre d'Offres par Date ({selected_country}, {selected_sector})", fontsize=14, loc='center')
    plt.xlabel("Date de Publication", fontsize=12)
    plt.ylabel("Nombre d'Offres", fontsize=12)
    plt.grid(True)
    st.pyplot(plt)

    # Tableau : Top 10 Pays
    st.markdown("##### Pays proposant le plus d'offres")
    top_countries = (
        filtered_data.groupby("Country")
        .agg(Total_Offers=("Job Title", "count"))
        .sort_values("Total_Offers", ascending=False)
        .head(10)
        .reset_index()
    )

    # Ajouter un style CSS pour centrer le tableau
    st.markdown(
        """
        <style>
        .dataframe-container {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Vérifier si le tableau contient des données
    if not top_countries.empty:
        # Envelopper le tableau dans un conteneur centré
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(top_countries, height=250)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Aucune donnée disponible pour les critères sélectionnés.")









# Section "Salaires"
elif menu == "Salaires":
    st.subheader("Analyse des Salaires par Métier")

    job_choice = st.selectbox("Choisissez un métier :", data["Job Title"].unique())
    filtered_data_job = data[data["Job Title"] == job_choice]

    if filtered_data_job.empty:
        st.warning("Aucune donnée disponible pour ce métier.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Distribution des Salaires Minimums")
            plt.figure(figsize=(5, 3))
            sns.histplot(filtered_data_job["Salary min"], kde=True, color="blue")
            plt.title("Distribution des Salaires Minimums", fontsize=10, loc='center')
            plt.xlabel("Salaire Minimum (€)", fontsize=8)
            plt.ylabel("Fréquence", fontsize=8)
            st.pyplot(plt)

        with col2:
            st.markdown("##### Salaires Minimums par Qualification")
            qualification_salary_min = (
                filtered_data_job.groupby('Qualifications')['Salary min'].mean().sort_values()
            )
            plt.figure(figsize=(5, 3))
            qualification_salary_min.plot(kind='barh', color='orange')
            plt.title("Salaires Min / Qualification", fontsize=10, loc='center')
            plt.xlabel("Salaire Minimum (€)", fontsize=8)
            plt.ylabel("Qualifications", fontsize=8)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            st.pyplot(plt)

        # Section : Boxplot des Salaires
        st.markdown("##### Répartition des Salaires Moyens (Boxplot)")
        salary_data_country = filtered_data_job.dropna(subset=["Country", "Average Salary"])

        if salary_data_country.empty:
                st.warning("Aucune donnée valide pour générer un boxplot.")
        else:
                plt.figure(figsize=(20, 8))  # Taille modérée
                sns.boxplot(data=salary_data_country, x="Country", y="Average Salary", palette="coolwarm")

                # Personnalisation du graphique
        plt.title("Répartition des Salaires par Pays", fontsize=22)
        plt.xlabel("Pays", fontsize=10)
        plt.ylabel("Salaire Moyen (€)", fontsize=10)
        plt.xticks(rotation=45, fontsize=12)
        st.pyplot(plt)

        # Section : Tableau des Salaires et Qualifications
        st.markdown("##### Détails des Salaires et Qualifications")
        grouped_data = filtered_data_job.groupby('Qualifications').agg({
                'Salary min': 'mean',
                'Salary max': 'mean',
                'Average Salary': 'mean'
            }).sort_values('Average Salary', ascending=False)

        grouped_data.columns = [
                'Salaire Minimum Moyen (€)',
                'Salaire Maximum Moyen (€)',
                'Salaire Moyen (€)'
            ]
        grouped_data.reset_index(inplace=True)

        if grouped_data.empty:
                st.warning("Aucune donnée disponible pour le tableau.")
        else:
                st.dataframe(grouped_data, height=300)

elif menu == "Moteur de recherche":

    # Barre de recherche
    # Assuming `data` is already loaded as a pandas DataFrame
    st.subheader("Recherche d'emplois")

    with st.form(key='search_form'):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            search_term = st.text_input("Compétences recherchées (séparées par des espaces)", "")

        with col2:
            # Liste unique des pays à partir des données
            countries = sorted(data["Country"].dropna().unique())

            # Liste déroulante pour sélectionner un pays
            selected_country = st.selectbox("Choisissez un pays :", ["Tous"] + countries)

        with col3:
            submit_search = st.form_submit_button(label='Explorer')

    if submit_search:
        # Diviser les mots-clés de recherche
        search_terms = search_term.split()

        # Construire un filtre restrictif pour les mots-clés
        skill_filter = data['skills'].apply(
            lambda x: all(term.lower() in x.lower() for term in search_terms) if pd.notna(x) else False
        )

        # Filtrer par localisation
        if selected_country != "Tous":
            location_filter = data['Country'].str.contains(selected_country, case=False, na=False)
        else:
            location_filter = True  # Match all rows

        # Appliquer les filtres
        filtered_data = data[skill_filter & location_filter]

        if not filtered_data.empty:
            st.success(f"{len(filtered_data)} emploi(s) trouvé(s)")

            with st.expander("Afficher les résultats"):
                result_columns = ['Job Title', 'Work Type', 'location', 'Preference', 'Qualifications', 'Job_Sector',
                                  'Salary min', 'Average Salary', 'Experience min', 'Job Posting Date',
                                  'Company_Sector', 'Company']
                # Display only available columns
                available_columns = [col for col in result_columns if col in filtered_data.columns]
                st.dataframe(filtered_data[available_columns])
        else:
            st.warning("Aucun emploi trouvé. Essayez avec d'autres mots-clés ou emplacements.")
        st.title("Analyse des Tendances des Compétences")

        # Convertir les chaînes en listes réelles et concaténer les compétences
        all_skills = " ".join(
            [skill for skills_list in filtered_data['skills'].apply(ast.literal_eval) for skill in skills_list]
        )

        # Générer un Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_skills)

        # Afficher le Word Cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)  # Affichage avec Streamlit au lieu de plt.show()

elif menu == "WordCloud":
    import re
    import nltk
    from nltk.corpus import stopwords
    from wordcloud import WordCloud, STOPWORDS
    from collections import Counter
    from unidecode import unidecode
    from nltk.stem import SnowballStemmer


    # Télécharger les stopwords en français
    nltk.download('stopwords')
    stop_words = set(stopwords.words('french'))

    # Stopwords personnalisés
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        "c'est", "cela", "plus", "autre", "tous", "toutes", "aussi", "être",
        "ces", "fait", "si", "dans", "sur", "comme", "ceux", "nouveau", "car"
    ])

    # Texte à analyser
    corpus = [
        "C'est un bouleversement mondial, et notre façon de vivre change plus vite que nous n'aurions jamais pu l'imaginer.",
    "Qui aurait pu prévoir que les entreprises qui ont des activités à distance seraient les survivantes du premier trimestre 2020?",
            "C'est un changement qu’aucune personne n’aurait pu voir venir et auquel de nombreuses entreprises n'étaient pas préparées.",
            "Le marché du travail a changé, et tandis que certaines opportunités disparaissent, d'autres font surface.",
            "Maintenant, tout repose sur la valeur.",
            "Ceux qui ont moins de valeur pour leur entreprise perdent leur emploi.",
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
            "Bien que ce changement n'ait pas été facile jusqu'à présent, nous continuons à apprendre des façons d'améliorer les choses.",
            "Les personnes refont leur emploi du temps pour correspondre aux nouvelles conditions de travail à distance.",
            "Maintenant, pour gagner de l'argent, vous avez besoin d'un ordinateur, d'une connexion Internet fiable, d'appareils intelligents, etc.",
            "Vous avez besoin de compétences générales en informatique par exemple, même si ce n’est pas votre domaine.",
            "La plupart des facteurs demandés n'étaient pas nécessaires dans le monde pré-Covid.",
            "Et il serait naïf de croire que le monde post-Covid redeviendra comme avant.",
            "Le flux de travail des entreprises a été facilité par la présence d'e-mails, de plateformes de visioconférence etc.",
            "Plusieurs entreprises admettent qu'elles ne reprendront pas le travail en physique lorsque la pandémie de COVID-19 serait terminée.",
            "Au cours de l'année, l'infrastructure Internet s'est développée à un niveau tel qu'elle peut répondre sans soucis à ces besoins.",
            "Un autre changement majeur est le fait que la période de travail de 8 heures sera effacée.",
            "Car les employés sont tous en ligne tout le temps grâce à certaines plateformes.",
            "Mais cela peut avoir un inconvénient, car il sera facile de perdre la notion du temps alloué au travail.",
            "C’est à l'employé de trouver le juste milieu entre sa vie de travail et personnelle."
    ]

    # Fonction pour supprimer les accents
    def remove_accents(text):
        return unidecode(text)

    # Étape 1 : Nettoyage
    def clean_corpus(corpus):
        # Convertir en minuscules et supprimer les accents
        corpus_cleaned = [remove_accents(doc.lower()) for doc in corpus]
        # Supprimer les chiffres et la ponctuation
        corpus_cleaned = [re.sub(r'\d+', '', doc) for doc in corpus_cleaned]
        corpus_cleaned = [re.sub(r'[^\w\s]', '', doc) for doc in corpus_cleaned]
        # Supprimer les mots très courts (par ex., "a", "le")
        corpus_cleaned = [re.sub(r'\b\w{1,2}\b', '', doc) for doc in corpus_cleaned]
        # Supprimer les stopwords
        corpus_cleaned = [' '.join([word for word in doc.split() if word not in stop_words]) for doc in corpus_cleaned]
        return corpus_cleaned

    # Étape 2 : Stemmatisation
    def stem_corpus(corpus):
        stemmer = SnowballStemmer('french')
        return [' '.join([stemmer.stem(word) for word in doc.split()]) for doc in corpus]

    # Générer le WordCloud
    def generate_wordcloud(corpus):
        word_freq = Counter(' '.join(corpus).split())
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=custom_stopwords,
            colormap='coolwarm',
            max_words=100,
            contour_width=1,
            contour_color='black'
        ).generate_from_frequencies(word_freq)
        return wordcloud

    # Intégration avec Streamlit
    st.title("WordCloud des tendances du marché du travail")
    st.markdown("Voici un WordCloud basé sur les tendances observées dans le marché du travail européen en 2021.")

    # Nettoyer et transformer le corpus
    corpus_cleaned = clean_corpus(corpus)
    corpus_stemmed = stem_corpus(corpus_cleaned)

    # Générer le WordCloud
    wordcloud = generate_wordcloud(corpus_stemmed)

    # Afficher le WordCloud sur Streamlit
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Tendances du marché du travail européen en 2021", fontsize=16)
    st.pyplot(fig)

    # Option pour télécharger le WordCloud
    st.markdown("### Téléchargez le WordCloud")
    st.download_button(
        label="Télécharger l'image",
        data=open("tendance_marche_travail_europe_2021.png", "rb").read(),
        file_name="tendance_marche_travail_europe_2021.png",
        mime="image/png"
    )
