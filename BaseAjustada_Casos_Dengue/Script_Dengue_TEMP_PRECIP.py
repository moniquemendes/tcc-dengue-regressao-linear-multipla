# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# 
# MONIQUE MENDES

# coding: utf-8

# In[0.1]: Instalação dos pacotes
!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install playsound
!pip install pingouin
!pip install emojis
!pip install statstests

# In[0.2]: Importação dos pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import plotly.graph_objects as go # gráficos 3D
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from sklearn.preprocessing import LabelEncoder # transformação de dados
from playsound import playsound # reprodução de sons
import pingouin as pg # outro modo para obtenção de matrizes de correlações
import emojis # inserção de emojis em gráficos
from statstests.process import stepwise # procedimento Stepwise
from statstests.tests import shapiro_francia # teste de Shapiro-Francia
from scipy.stats import boxcox # transformação de Box-Cox
from scipy.stats import norm # para plotagem da curva normal
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'


# In[0.3]: Importando Banco de Dados
#############################################################################
#                          REGRESSÃO LINEAR MÚLTIPLA                           #
#                        CARREGAMENTO DA BASE DE DADOS                 #
#############################################################################
#IMPORTAR A BASE DO ANO - 2020
#FONTE: ADAPTADO A BASE DO SNIS/SINAN/TABNET-SAÚDE/IBGE
df_dengue =  pd.read_excel('Base_2020_TEMP_PRECIP.xlsx')
#df_dengue =  pd.read_excel('Base_2021.xlsx')
#df_dengue =  pd.read_excel('Base_2022.xlsx')
df_dengue

# Transformação do 'cod_ibge' para o tipo 'str'
df_dengue['cod_ibge'] = df_dengue['cod_ibge'].astype('str')

# Características das variáveis do dataset
df_dengue.info()

# Estatísticas univariadas
df_dengue.describe()

#############################################################################
#ESTATÍSTICAS DESCRITIVAS DOS DADOS
 
df_dengue[['casos_dengue','precipitacao_total_anual','temp_media_anual','total_coleta_residuos','desp_saude_percapta','dens_demografica']].describe()

df_dengue['saneamento_basico'].value_counts().sort_index()
df_dengue['abastecimento_agua'].value_counts().sort_index()
df_dengue['esgotamento_sanitario'].value_counts().sort_index()
df_dengue['coleta_residuos'].value_counts().sort_index()
df_dengue['drenagem_urbana'].value_counts().sort_index()



#%% Análise do coeficiente de correlação de Pearson entre as variáveis quantitativas (Investigação)

pg.rcorr(df_dengue[['casos_dengue','precipitacao_total_anual','temp_media_anual','vento_vlc_media_anual','total_coleta_residuos', 'pib_municipio','desp_saude_percapta','dens_demografica' ]],
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#####################################################################################################################

#OBTER DUMMIES DAS VARIÁVEIS CATEGÓRICAS -IGUAL A FALSE (0- Sim/ 1 - Não)

# df_dengue_dummies = pd.get_dummies(df_dengue,
#                         columns = ['saneamento_basico',
#                                     'abastecimento_agua',
#                                     'esgotamento_sanitario',
#                                     'coleta_residuos',
#                                     'drenagem_urbana'],
#                         dtype = int,
#                         drop_first = True)

##############################################################################
#FAZER O MAPA DE CALOR.......

###############################################################################
# # # #Correlação com todas as variaveis qualitativas dumizadas 

# import pingouin as pg

# correlation_matrix2 = pg.rcorr(df_dengue, method='pearson',
#                               upper='pval', decimals=4,
#                               pval_stars={0.01: '***',
#                                           0.05: '**',
#                                           0.10: '*'})
# correlation_matrix2

###############################################################################
#Remover Observações com valores nulo (Casos de dengue) 

# df_dengue_dummies = df_dengue_dummies.dropna()

###############################################################################
# # #  # MODELO DE REGRESSÃO LINEAR MÚLTIPLA (MQO) - SEM LOG DAS VARIAVEIS

# Estimação do Modelo - Variaveis Sim 
# modelo_dengue = sm.OLS.from_formula( 'casos_dengue ~  + \
#                                       precipitacao_total_anual  + temp_media_anual + \
#                                       total_coleta_residuos + dens_demografica + pib_municipio + \
#                                       desp_saude_percapta + desp_percapta_RSU + coleta_residuos_Sim + \
#                                       saneamento_basico_Sim + drenagem_urbana_Sim',
#                                     ##  esgotamento_sanitario_Sim + abastecimento_agua_Sim',
#                                       df_dengue_dummies).fit()
    
# #OBTENÇÃO DOS OUTPUTS
# modelo_dengue.summary()

###############################################################################
#%% Teste de verificação da aderência dos resíduos à normalidade

# Elaboração do teste de Shapiro-Francia
# teste_sf = shapiro_francia(modelo_dengue.resid)
# round(teste_sf['p-value'], 5)

# # Tomando a decisão por meio do teste de hipóteses

# alpha = 0.05 # nível de significância do teste

# if teste_sf['p-value'] > alpha:
#  	print('Não se rejeita H0 - Distribuição aderente à normalidade')
# else:
#  	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

###############################################################################
#%% Histograma dos resíduos do modelo OLS

# # Parâmetros de referência para a distribuição normal teórica
# (mu, std) = norm.fit(modelo_dengue.resid)

# # Criação do gráfico
# plt.figure(figsize=(15,10))
# plt.hist(modelo_dengue.resid, bins=35, density=True, alpha=0.7, color='purple')
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 1000)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, linewidth=3, color='red')
# plt.title('Resíduos do Modelo', fontsize=20)
# plt.xlabel('Resíduos', fontsize=22)
# plt.ylabel('Frequência', fontsize=22)
# plt.show()

###############################################################################
#%% Realizando a transformação de Box-Cox na variável dependente

# y_box, lmbda = boxcox(df_dengue_dummies['casos_dengue'])

# # Valor obtido para o lambda
# print(lmbda)

# # Adicionando ao banco de dados
# df_dengue_dummies['casos_dengue_bc'] = y_box

# ###############################################################################
# # Estimação do Modelo - Variaveis Sim 
# modelo_dengue_bc = sm.OLS.from_formula( 'casos_dengue_bc ~  + \
#                                         precipitacao_total_anual  + temp_media_anual + \
#                                         total_coleta_residuos + dens_demografica + pib_municipio + \
#                                         desp_saude_percapta + desp_percapta_RSU + coleta_residuos_Sim + \
#                                         saneamento_basico_Sim + drenagem_urbana_Sim + \
#                                         esgotamento_sanitario_Sim + abastecimento_agua_Sim',
#                                       df_dengue_dummies).fit()
    
# #OBTENÇÃO DOS OUTPUTS
# modelo_dengue_bc.summary()

###############################################################################
#%% Reavaliando aderência à normalidade dos resíduos do modelo

# Teste de Shapiro-Francia
# teste_sf_bc = shapiro_francia(modelo_dengue_bc.resid)

# # Tomando a decisão por meio do teste de hipóteses

# alpha = 0.05 # nível de significância do teste

# if teste_sf_bc['p-value'] > alpha:
#  	print('Não se rejeita H0 - Distribuição aderente à normalidade')
# else:
#  	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
    
#%% Removendo as variáveis que não apresentam significância estatística

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

# # Stepwise do modelo
# modelo_stepwise_bc = stepwise(modelo_dengue_bc, pvalue_limit=0.05)

# # Teste de Shapiro-Francia
# teste_sf_step = shapiro_francia(modelo_stepwise_bc.resid)

#%% Novo histograma dos resíduos do modelo

# # Parâmetros de referência para a distribuição normal teórica
# (mu_bc, std_bc) = norm.fit(modelo_stepwise_bc.resid)

# # Criação do gráfico
# plt.figure(figsize=(15,10))
# plt.hist(modelo_stepwise_bc.resid, bins=30, density=True, alpha=0.8, color='darkblue')
# xmin_bc, xmax_bc = plt.xlim()
# x_bc = np.linspace(xmin_bc, xmax_bc, 1000)
# p_bc = norm.pdf(x_bc, mu_bc, std_bc)
# plt.plot(x_bc, p_bc, linewidth=3, color='red')
# plt.title('Resíduos do Modelo Box-Cox', fontsize=20)
# plt.xlabel('Resíduos', fontsize=22)
# plt.ylabel('Frequência', fontsize=22)
# plt.show()


#%% Criação da função para o teste de Breusch-Pagan (heterocedasticidade)

# def breusch_pagan_test(modelo):

#     df = pd.DataFrame({'yhat':modelo.fittedvalues,
#                         'resid':modelo.resid})
   
#     df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
#     modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
#     anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
#     anova_table['sum_sq'] = anova_table['sum_sq']/2
    
#     chisq = anova_table['sum_sq'].iloc[0]
   
#     p_value = stats.chi2.pdf(chisq, 1)*2
    
#     print(f"chisq: {chisq}")
    
#     print(f"p-value: {p_value}")
    
#     return chisq, p_value


#%% Analisando a presença de heterocedasticidade no modelo

# teste_bp_original = breusch_pagan_test(modelo_dengue_bc)

# # Tomando a decisão por meio do teste de hipóteses

# alpha = 0.05 # nível de significância do teste

# if teste_bp_original[1] > alpha:
#     print('Não se rejeita H0 - Ausência de Heterocedasticidade')
# else:
#  	print('Rejeita-se H0 - Existência de Heterocedasticidade')
    
###############################################################################
#VIF (Variance Inflation Factor) é usado para detectar colinearidade  entre as 
#variaveis explicativas. Se  VIF de uma variável for muito alto (geralmente acima de 5 ou 10)
# significa que ela está altamente correlacionada com outras variáveis do modelo,
# o que pode distorcer as estimativas dos coeficientes.

# from statsmodels.stats.outliers_influence import variance_inflation_factor

# # Selecionar apenas as variáveis independentes do modelo
# X = df_dengue_dummies[['precipitacao_total_anual', 'temp_media_anual', 'vento_vlc_media_anual', 
#                         'total_coleta_residuos', 'desp_percapta_RSU', 'coleta_residuos_Sim']]

# # Adicionar a constante ao modelo (intercepto)
# X = sm.add_constant(X)

# # Calcular o VIF para cada variável
# vif_data = pd.DataFrame()
# vif_data["Variável"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# # Exibir os resultados
# print(vif_data)

##############################################################################



