import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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


# Carregar o dataset
df = pd.read_csv('College_Data.csv', index_col=0)
df['Private'] = df['Private'].apply(lambda x: 1 if x == 'Yes' else 0)
df.head()

df.describe()

# Existe um outlier que possui uma taxa de graduação muito alta além do normal, vamos excluí-lo
college = df[df['Grad.Rate'] > 100]
df = df.drop(college.index)

# Verificar a relação de todas as variáveis com a variável target
# Dessa forma é possível perceber quais variáveis podem ser padronizadas ou normalizadas.
# Levando em consideração que variáveis que apresentam uma distribuição normal serão padronizadas e as que não apresentam serão normalizadas.
#sns.pairplot(df, hue='Private')

# Separando as variáveis que serão padronizadas e as que serão normalizadas
df_standard = df[['Top10perc', 'Top25perc', 'Outstate', 'Room.Board', 'S.F.Ratio', 'perc.alumni', 'Private']]
df_normalized = df.drop(['Top10perc', 'Top25perc', 'Outstate', 'Room.Board', 'S.F.Ratio', 'perc.alumni',], axis=1)

#Aplicando a correlação para eliminar variáveis que não possuem relação com a variável target
#sns.heatmap(df_standard.corr(),annot=True)
df_standard = df_standard.drop(['Private', 'Outstate', 'S.F.Ratio', 'perc.alumni'], axis=1)


#sns.heatmap(df_normalized.corr(),annot=True)
df_normalized = df_normalized.drop(['Apps', 'Accept', 'P.Undergrad', 'Private'], axis=1)

# Verificando se há relação entre as despesas e a taxa de graduação
# É possível concluir que alunos que possuem altas despesas tendem a se formar mais, porém, a relação não é tão forte.
#sns.lmplot(x='Expend', y='Grad.Rate', data=df, hue='Private', palette='coolwarm')



# Obs. Não foi aplicado as etapas de normalização e padronização, pois o modelo teve uma acurácia inferior em relação a quando não foi aplicado.
# Pré-processamento, aplicando a padronização e normalização
#normalizer = MinMaxScaler()
#scaler = StandardScaler()

# df_standard = scaler.fit_transform(df_standard)
# df_normalized = normalizer.fit_transform(df_normalized)

# Depois de padronizar e normalizar, é necessário juntar os dois dataframes
df_standard = pd.DataFrame(df_standard, columns=df_standard.columns, index=df_standard.index)
df_normalized = pd.DataFrame(df_normalized, columns=df_normalized.columns, index=df_normalized.index)
x = pd.concat([df_standard, df_normalized], axis=1)

# Variável target
y = pd.Series(df['Private'])


# Utilizando o modelo KMeans para clusterizar os dados
# Gerar dois clusters, um para universidades privadas e outro para universidades públicas
kmeans = KMeans(n_clusters=2, random_state=15)
model = kmeans.fit(x)

# Centros de cada variável
print(kmeans.cluster_centers_)


# O modelo obteve uma acurácia de 83%
clusters = model.predict(x)
print(accuracy_score(clusters, y))
