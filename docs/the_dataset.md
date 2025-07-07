## Descrição do Dataset
<p style= "text-align: justify">
O conjunto de dados utilizado neste projeto foi obtido na plataforma Kaggle, disponibilizado pelo autor Marshal Patel. Este dataset é composto por informações biomédicas e características de pacientes, sendo fundamental para a criação de modelos de predição de diabetes. O dataset pode ser acessado clicado nesse link <a href="https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset">Link do Dataset</a>:</p>

## Descrição dos Parâmetros

Abaixo, segue uma descrição detalhada de cada um dos parâmetros (colunas) presentes no conjunto de dados:

<ul style= "text-align: justify">
    <li><strong>Gender:</strong> O gênero do paciente (Feminino ou Masculino). </li>
    <li><strong>AGE:</strong> A idade do paciente em anos. </li>
    <li><strong>Urea:</strong> O nível de ureia no sangue, um indicador da função renal. </li>
    <li><strong>Cr (Creatinina):</strong> O nível de creatinina no sangue, outro importante indicador da função dos rins. </li>
    <li><strong>HbA1c (Hemoglobina Glicada):</strong> Um exame que mede a média dos níveis de açúcar no sangue nos últimos 2 a 3 meses. É um parâmetro chave para o diagnóstico e monitoramento do diabetes. </li>
    <li><strong>Chol (Colesterol):</strong>O nível de colesterol total no sangue.</li>
    <li><strong>TG (Triglicerídeos):</strong>O nível de triglicerídeos no sangue, um tipo de gordura presente na corrente sanguínea.</li>
    <li><strong>HDL (Lipoproteína de Alta Densidade):</strong>Conhecido como "colesterol bom", ajuda a remover o excesso de colesterol do corpo.</li>
    <li><strong>LDL (Lipoproteína de Baixa Densidade):</strong>Conhecido como "colesterol ruim", seu acúmulo pode levar à formação de placas nas artérias.</li>
    <li><strong>VLDL (Lipoproteína de Muito Baixa Densidade):</strong>Um precursor do LDL, também associado ao acúmulo de placas nas artérias.</li>
    <li><strong>BMI (Índice de Massa Corporal):</strong>Uma medida da gordura corporal baseada na altura e no peso do indivíduo.</li>
    <li><strong>CLASS:</strong>A classe de diagnóstico do paciente, podendo ser <strong>N</strong> (Não diabético), <strong>P</strong> (Pré-diabético) ou <strong>Y</strong> (Diabético).</li>
</ul>
<p style= "text-align: justify">
Este dataset foi utilizado em todas as técnicas do meu projeto como base para explorar as possibilidades.</p>

# Análise de Dados e Preparação para Modelagem de Predição de Diabetes

Aqui eu faço uma descrição que descreve o processo completo de análise e pré-processamento de um dataset de predição de diabetes. O objetivo é preparar os dados de forma robusta, criando uma base sólida e confiável para o treinamento de diferentes algoritmos de Machine Learning capazes de classificar pacientes como não-diabéticos (N), pré-diabéticos (P) ou diabéticos (Y).

## 1\. Aquisição e Exploração dos Dados

O primeiro passo em qualquer projeto de ciência de dados é obter e entender o conjunto de dados com o qual estamos trabalhando.

### 1.1. Carregamento do Dataset

Utilizei a biblioteca `opendatasets` para baixar o conjunto de dados diretamente da plataforma Kaggle e o `pandas` para carregá-lo em um DataFrame, que é a estrutura de dados fundamental para análise em Python.

```python
import opendatasets as od
import pandas as pd

od.download("https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset")
```

### 1.2. Análise Exploratória Inicial

Com os dados carregados, realizei uma análise exploratória inicial para entender sua estrutura, tipos de dados e a presença de valores ausentes.

  - **`.head()`**: Visualizei as 5 primeiras linhas para ter uma primeira impressão dos dados e das colunas.

```python
diabetes_dataset.head(5)
```

```
   ID  No_Pation Gender  AGE  Urea  Cr  HbA1c  Chol   TG  HDL  LDL  VLDL   BMI CLASS
0  502      17975      F   50   4.7  46    4.9   4.2  0.9  2.4  1.4   0.5  24.0     N
1  735      34221      M   26   4.5  62    4.9   3.7  1.4  1.1  2.1   0.6  23.0     N
2  420      47975      F   50   4.7  46    4.9   4.2  0.9  2.4  1.4   0.5  24.0     N
3  680      87656      F   50   4.7  46    4.9   4.2  0.9  2.4  1.4   0.5  24.0     N
4  504      34223      M   33   7.1  46    4.9   4.9  1.0  0.8  2.0   0.4  21.0     N
```

  - **`.info()`**: Verifiquei os tipos de dados de cada coluna e a contagem de valores não-nulos. Felizmente, o dataset não apresentou valores ausentes, o que simplifica a etapa de limpeza.

```python
diabetes_dataset.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 14 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   ID         1000 non-null   int64
 1   No_Pation  1000 non-null   int64
 2   Gender     1000 non-null   object
 3   AGE        1000 non-null   int64
 4   Urea       1000 non-null   float64
 5   Cr         1000 non-null   int64
 6   HbA1c      1000 non-null   float64
 7   Chol       1000 non-null   float64
 8   TG         1000 non-null   float64
 9   HDL        1000 non-null   float64
 10  LDL        1000 non-null   float64
 11  VLDL       1000 non-null   float64
 12  BMI        1000 non-null   float64
 13  CLASS      1000 non-null   object
```

## 2\. Pré-processamento e Limpeza dos Dados

A qualidade do modelo de Machine Learning depende diretamente da qualidade dos dados de entrada. Nesta etapa, corrigi inconsistências e transformei os dados para um formato adequado para o treinamento.

### 2.1. Limpeza de Dados Categóricos

identifiquei que as colunas categóricas `Gender` e `CLASS` continham ruídos, como valores diferentes para a mesma categoria (ex: 'N' e 'N ') e inconsistência de maiúsculas/minúsculas (ex: 'F' e 'f').

```python
# Verificando os valores únicos antes da limpeza
list_class = diabetes_dataset['CLASS'].unique()
list_gender = diabetes_dataset['Gender'].unique()
print(f"For label CLASS: {list_class}\nFor label Gender: {list_gender}")
# Saída:
# For label CLASS: ['N' 'N ' 'P' 'Y' 'Y ']
# For label Gender: ['F' 'M' 'f']
```

Para corrigir, apliquei uma função para converter todos os valores para maiúsculas e remover espaços em branco no início e no fim das strings.

```python
# Aplicando a limpeza
diabetes_dataset['CLASS'] = diabetes_dataset['CLASS'].apply(lambda x: str.upper(x).strip())
diabetes_dataset['Gender'] = diabetes_dataset['Gender'].apply(lambda x: str.upper(x))

# Verificando o resultado após a limpeza
list_class = diabetes_dataset['CLASS'].unique()
list_gender = diabetes_dataset['Gender'].unique()
print(f"For label CLASS: {list_class}\nFor label Gender: {list_gender}")
# Saída:
# For label CLASS: ['N' 'P' 'Y']
# For label Gender: ['F' 'M']
```

### 2.2. Remoção de Colunas Irrelevantes

As colunas `ID` e `No_Pation` são identificadores únicos para cada paciente. Elas não possuem valor preditivo e podem ser consideradas ruído para o modelo. Por isso, foram removidas.

```python
diabetes_dataset = diabetes_dataset.drop(columns=['ID', 'No_Pation'])
```

### 2.3. Codificação de Variáveis Categóricas

Modelos de machine learning requerem que todas as features de entrada sejam numéricas. Portanto, converti as colunas categóricas `Gender` e `CLASS`. Obs.: Em determinados algoritmos, essa parte não interfere tanto mas deixei a menção para quando for usada nos algoritmos.

  - **Gender (One-Hot Encoding)**: Para a coluna `Gender`, que é uma variável nominal (não há uma ordem intrínseca entre 'M' e 'F'), utilizei a técnica **One-Hot Encoding** com `pd.get_dummies`. Isso cria novas colunas binárias (`Gender_F` e `Gender_M`), evitando que o modelo atribua um peso ordinal indevido.

```python
df_diabetes = pd.get_dummies(diabetes_dataset, columns=['Gender'], dtype=int)
```

  - **CLASS (Label Encoding)**: A coluna `CLASS` é a nossa variável-alvo (target). Utilizei o `LabelEncoder` da biblioteca `scikit-learn` para transformá-la em valores numéricos (0, 1, 2). Essa é uma abordagem padrão para a variável de saída em problemas de classificação.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_diabetes['CLASS'] = le.fit_transform(df_diabetes['CLASS'])

# Verificando o mapeamento das classes
# Saída: Class Map: [(0, 'N'), (1, 'P'), (2, 'Y')]
```

## 3\. Preparação dos Dados para a Modelagem

Antes de treinar qualquer modelo, é crucial preparar os dados corretamente, tratando questões como desbalanceamento de classes e a escala das features.

### 3.1. Visualização da Distribuição das Classes

Plotei um gráfico para visualizar a distribuição das classes e identificamos um **forte desbalanceamento**. A classe 'Y' (diabético) era massivamente majoritária, o que pode enviesar o modelo a prever sempre essa classe, ignorando as minoritárias ('N' e 'P').

### 3.2. Divisão em Dados de Treino e Teste

Dividi o dataset em conjuntos de treino (80%) e teste (20%). Usamos o parâmetro `stratify=y` para garantir que a proporção das classes nos conjuntos de treino e teste fosse a mesma do dataset original, o que é fundamental em casos de dados desbalanceados.

```python
from sklearn.model_selection import train_test_split
X = df_diabetes.drop('CLASS', axis=1)
y = df_diabetes['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

### 3.3. Balanceamento das Classes com SMOTE

Para lidar com o desbalanceamento, apliquei a técnica **SMOTE (Synthetic Minority Over-sampling Technique)**. O SMOTE cria novas amostras sintéticas das classes minoritárias ('N' e 'P') no conjunto de **treino**, baseando-se nas amostras existentes. Isso permite que o modelo aprenda as características de todas as classes de forma mais equilibrada.

**Importante**: O SMOTE foi aplicado **apenas nos dados de treino** para evitar vazamento de dados (data leakage), garantindo que o conjunto de teste permaneça uma representação real e "inédita" do problema.

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

print("Distribuição antes do SMOTE:", Counter(y_train))
# Saída: Class distribution before SMOTE: Counter({2: 675, 0: 82, 1: 43})

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nDistribuição após o SMOTE:", Counter(y_train_resampled))
# Saída: Class Distribution after SMOTE: Counter({2: 675, 0: 675, 1: 675})
```

### 3.4. Escalonamento das Features (Feature Scaling)

Muitos algoritmos de Machine Learning (como Regressão Logística, SVM, Redes Neurais, etc.) são sensíveis à escala das features. Uma feature com uma grande amplitude de valores (ex: `Cr`) poderia dominar outras com valores menores (ex: `Urea`), distorcendo o aprendizado. Para evitar isso, usei o `StandardScaler`, que padroniza as features para que tenham média 0 e desvio padrão 1.

O `scaler` foi treinado (`fit_transform`) com os dados de treino e depois apenas aplicado (`transform`) nos dados de teste, novamente para evitar vazamento de informações do conjunto de teste para o modelo.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
```

## 4\. Conclusão do Pré-Processamento e Próximos Passos

Ao final desta detalhada etapa de pré-processamento, os dados estão limpos, balanceados e padronizados, prontos para serem utilizados na fase de modelagem. As principais transformações realizadas foram:

  - **Limpeza e Padronização**: Correção de inconsistências nos dados categóricos.
  - **Codificação de Features**: Conversão de variáveis categóricas em formato numérico.
  - **Balanceamento de Classes**: Mitigação do viés da classe majoritária usando SMOTE.
  - **Escalonamento de Features**: Normalização da escala das variáveis para um tratamento justo pelos algoritmos.

Com esta base sólida e confiável, os próximos passos envolvem treinar e avaliar diferentes algoritmos utilizando os conjuntos `X_train_scaled`, `y_train_resampled`, `X_test_scaled` e `y_test` (para casos onde esses subconjuntos precisam ser normalizados e bem distribuídos) para determinar qual modelo oferece a melhor performance para este problema.