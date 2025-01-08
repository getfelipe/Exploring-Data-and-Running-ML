import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Coluna Privada: Fator com níveis Sim ou Não indicando se a universidade é privada ou não
# Coluna Apps: Número de candidaturas recebidas
# Coluna Accept: Número de candidaturas aceitas
# Enroll: Número de estudantes matriculados
# Top10perc: Percentual de novos estudantes vindo do grupo dos 10% melhores do ensino médio
# Top25perc: Percentual de novos estudantes vindo do grupo dos 25% melhores do ensino médio
# F.Undergrad: Número de estudantes de graduação em tempo integral
# P.Undergrad: Número de estudantes de graduação em tempo parcial
# Outstate: Aulas para estudantes fora do estado
# Room.Board: Custo do alojamento
# Books: Custo de livros estimado
# Personal: Estimativa de gastos pessoais
# PhD: Percentual de professores com doutorado
# Terminal: Percentual da faculdade com graduação
# S.F.Ratio: Taxa de estudantes por faculdade
# perc.alumni: Percentual de ex-alunos que doam
# Expend: Despesas da faculdade por aluno
# Grad.Rate: Taxa de graduação

# Load the data
df = pd.read_csv('College_Data.csv', index_col=0)
df.head()

# Exploratory Data Analysis
df.describe()

# Scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column
sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', palette='coolwarm')

# Scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column
sns.lmplot(x='F.Undergrad', y='Outstate', data=df, hue='Private', palette='coolwarm')

# Stacked histogram showing Out of State Tuition based on the Private column using [sns.FacetGrid] and [plt.hist] methods.

df[df['Private'] == 'Yes']['Outstate'].plot(kind='hist', alpha=0.6, label='Private')
df[df['Private'] == 'No']['Outstate'].plot(kind='hist', alpha=0.6, label='Public')
plt.legend()


df[df['Private'] == 'Yes']['Grad.Rate'].plot(kind='hist', alpha=0.6, label='Private')
df[df['Private'] == 'No']['Grad.Rate'].plot(kind='hist', alpha=0.6, label='Public')
plt.legend()

# There seems to be a private school with a graduation rate of higher than 100%. What is the name of that school?
df[df['Grad.Rate'] > 100]

# Set that school's graduation rate to 100 so it makes sense
df['Grad.Rate']['Cazenovia College'] = 100


# K Means Cluster Creation
# Create an instance of a K Means model with 2 clusters

kmeans = KMeans(n_clusters=2)
df.drop('Private', axis=1, inplace=True)
kmeans.fit(df)

# Cluster center vectors
print(kmeans.cluster_centers_)

# Evaluation
# Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school
df['Cluster'] = df['Private'].apply(lambda x: 1 if x == 'Yes' else 0)

# Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))
