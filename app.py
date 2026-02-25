import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Configuração da página
st.set_page_config(page_title="Portfolio IA - Águia Sistemas", layout="wide")

# --- SIMULAÇÃO DE CARREGAMENTO DO MODELO ---
# (Aqui usamos os parâmetros que você definiu no seu treino)
def train_model_desafio_1():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['region'] = df['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)
    
    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y) 
    return model

# --- INTERFACE ---
st.title("🚀 Painel de Soluções em IA")
st.markdown("Apresentação dos desafios técnicos desenvolvidos para o processo seletivo.")

# Menu Lateral
aba = st.sidebar.selectbox("Escolha o Desafio", 
                         ["Home", "1. Previsão de Custos Médicos", "2. Titanic", "3. Churn", "4. Fraude"])

if aba == "Home":
    st.header("Bem-vindo ao meu Portfólio de Dados")
    st.write("Neste site interativo, você pode testar os modelos de Machine Learning desenvolvidos.")
    st.info("Selecione um desafio no menu lateral para começar.")

elif aba == "1. Previsão de Custos Médicos":
    st.header("🏥 Previsão de Custos de Seguro de Saúde")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Simulador de Previsão")
        st.write("Insira os dados do novo cliente:")
        
        # Inputs que batem com seu código
        age = st.number_input("Idade", min_value=18, max_value=100, value=45)
        sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Feminino" if x==0 else "Masculino")
        bmi = st.slider("IMC (Índice de Massa Corporal)", 10.0, 50.0, 26.0)
        children = st.selectbox("Número de Filhos", [0, 1, 2, 3, 4, 5], index=2)
        smoker = st.radio("Fumante?", options=[0, 1], format_func=lambda x: "Não" if x==0 else "Sim")
        region = st.selectbox("Região", options=[1, 2, 3, 4], format_func=lambda x: {1:"Sudoeste", 2:"Sudeste", 3:"Noroeste", 4:"Nordeste"}[x])
        
        if st.button("Calcular Custo Previsto"):
            model = train_model_desafio_1()
            input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                                    columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
            prediction = model.predict(input_data)
            st.success(f"O custo estimado para este perfil é: **R$ {prediction[0]:,.2f}**")

    with col2:
        st.subheader("Explicação Técnica")
        st.write("O modelo utiliza um algoritmo de **Random Forest** (Floresta Aleatória).")
        
        # Gráfico de Importância das Variáveis (conforme seu código)
        importances_data = {
            "Variável": ["Fumante", "IMC", "Idade", "Filhos", "Região", "Sexo"],
            "Importância": [0.6188, 0.2113, 0.1306, 0.0194, 0.0138, 0.0062]
        }
        st.bar_chart(pd.DataFrame(importances_data).set_index("Variável"))
        st.write("Note que o tabagismo é o fator preponderante no custo.")

# (As outras abas vamos preencher conforme você enviar os códigos 2, 3 e 4)
# --- FUNÇÃO DE TREINO DESAFIO 2 (Adicionar junto com as outras funções de treino) ---
def train_model_desafio_2():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})
    y = df['Survived']
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    return model

# --- DENTRO DA ESTRUTURA DE ABAS (Atualizar o menu lateral e as condições) ---

# No menu lateral, o selectbox já tem "2. Titanic"
if aba == "2. Titanic":
    st.header("🚢 Simulador de Sobrevivência - Titanic")
    st.markdown("Este modelo utiliza **Regressão Logística** para prever a probabilidade de sobrevivência com base em fatores socioeconômicos.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Crie seu Passageiro")
        pclass = st.selectbox("Classe Social (1ª é a mais alta)", [1, 2, 3])
        sex = st.radio("Sexo", options=[0, 1], format_func=lambda x: "Feminino" if x==0 else "Masculino")
        age = st.slider("Idade", 0, 100, 25)
        sibsp = st.number_input("Irmãos ou Cônjuge a bordo", 0, 10, 0)
        parch = st.number_input("Pais ou Filhos a bordo", 0, 10, 0)
        fare = st.number_input("Preço da Passagem (Fare)", 0.0, 600.0, 32.0)

        if st.button("Prever Sobrevivência"):
            model_titanic = train_model_desafio_2()
            input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], 
                                    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
            prediction = model_titanic.predict(input_data)
            
            if prediction[0] == 1:
                st.success("✅ O modelo prevê que esta pessoa **SOBREVIVERIA**.")
            else:
                st.error("❌ O modelo prevê que esta pessoa **NÃO SOBREVIVERIA**.")

    with col2:
        st.subheader("Insights dos Dados")
        st.write("Durante a análise, observamos padrões claros que o modelo aprendeu:")
        
        # Gráfico de Classe Social que você fez no código
        st.write("**Taxa de Sobrevivência por Classe:**")
        chart_data = pd.DataFrame({
            'Classe': ['1ª Classe', '2ª Classe', '3ª Classe'],
            'Taxa de Sobrevivência': [0.63, 0.47, 0.24] # Valores aproximados do Titanic original
        })
        st.bar_chart(chart_data.set_index('Classe'))
        
        st.info("💡 **Destaque:** Mulheres e passageiros da 1ª classe tiveram as maiores chances de sobrevivência, refletindo a política de 'mulheres e crianças primeiro' e a localização das cabines.")
        # --- FUNÇÃO DE TREINO DESAFIO 3 ---
def train_model_desafio_3():
    # Simulando o carregamento e preparação rápida
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    # Padronizando para o seu formato de códigos
    df['Contract_Code'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['Security_Code'] = df['Online Security'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['Support_Code'] = df['Tech Support'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    X = df[['tenure', 'MonthlyCharges', 'Contract_Code', 'Security_Code', 'Support_Code']]
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- ABA DO DESAFIO 3 ---
if aba == "3. Churn":
    st.header("📉 Retenção de Clientes (Churn Telecom)")
    st.subheader("Análise de Risco e Impacto Financeiro")

    # Cards de Resumo de Negócio (Baseado nos seus resultados)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prejuízo Mensal (Total)", "$ 139,130.85", delta="-15% vs mês anterior", delta_color="inverse")
    col_b.metric("Ticket Médio de Perda", "$ 74.44")
    col_c.metric("Principal Motivo", "Suporte Técnico")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Simulador de Risco de Cancelamento")
        tenure = st.slider("Meses de Contrato (Tenure)", 0, 72, 12)
        monthly = st.number_input("Valor da Mensalidade ($)", 18.0, 120.0, 70.0)
        contract = st.selectbox("Tipo de Contrato", options=[0, 1, 2], 
                                format_func=lambda x: {0:"Mensal", 1:"Anual", 2:"Bienal"}[x])
        security = st.radio("Possui Segurança Online?", [1, 0], format_func=lambda x: "Sim" if x==1 else "Não")
        support = st.radio("Possui Suporte Técnico?", [1, 0], format_func=lambda x: "Sim" if x==1 else "Não")

        if st.button("Avaliar Risco de Churn"):
            model_churn = train_model_desafio_3()
            # Ajustando nomes de colunas para o modelo
            input_df = pd.DataFrame([[tenure, monthly, contract, security, support]], 
                                    columns=['tenure', 'MonthlyCharges', 'Contract_Code', 'Security_Code', 'Support_Code'])
            prob = model_churn.predict_proba(input_df)[0][1]
            
            if prob > 0.5:
                st.warning(f"⚠️ **ALTO RISCO:** Probabilidade de cancelamento de {prob:.2%}")
                st.write("👉 **Sugestão:** Oferecer upgrade para contrato anual e suporte técnico gratuito.")
            else:
                st.success(f"✅ **BAIXO RISCO:** Probabilidade de cancelamento de {prob:.2%}")

    with col2:
        st.subheader("Por que os clientes saem?")
        # Gráfico dos Top Motivos (Baseado no seu código)
        motivos = pd.DataFrame({
            'Motivo': ['Competitor made better offer', 'Attitude of support person', 'Attitude of service provider'],
            'Clientes': [311, 220, 135]
        })
        st.bar_chart(motivos.set_index('Motivo'))
        
        st.info("💡 **Insight:** O impacto do suporte técnico é crítico. Clientes sem suporte têm 3x mais chance de cancelar.")
        # --- FUNÇÃO DE TREINO DESAFIO 4 ---
def train_model_desafio_4():
    # Usando uma amostra para o treino ser rápido no deploy
    url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    df = pd.read_csv(url)
    
    fraudes = df[df['Class'] == 1]
    normais = df[df['Class'] == 0].sample(n=492, random_state=42)
    df_balanceado = pd.concat([fraudes, normais])
    
    X = df_balanceado.drop(['Class'], axis=1)
    y = df_balanceado['Class']
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

# --- ABA DO DESAFIO 4 ---
if aba == "4. Fraude":
    st.header("🛡️ Detecção de Fraude em Cartões")
    st.subheader("Classificação com Dados Desbalanceados")

    # Resumo de Métricas Críticas
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Recall (Sensibilidade)", "92%", "Foco: Capturar Fraudes")
    col_m2.metric("Precisão", "99%", "Evita Falsos Alarmes")
    col_m3.metric("F1-Score", "0.95")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Testar Detecção em Tempo Real")
        st.write("Ajuste as variáveis principais para testar o comportamento do modelo:")
        
        # Simulando entradas simplificadas (Baseadas nas colunas V1, V2... do dataset)
        v1 = st.slider("V1 (Intensidade da Anomalia)", -30.0, 5.0, 0.0, help="Valores muito negativos costumam indicar desvio do padrão de consumo.")
        v4 = st.slider("V4 (Nível de Risco Estático)", -5.0, 15.0, 1.0, help="Valores altos nesta componente estão fortemente correlacionados a fraudes neste modelo.")
        amount = st.number_input("Valor da Transação ($)", 0.0, 5000.0, 122.0)
        
        if st.button("Analisar Transação"):
            model_fraude, cols = train_model_desafio_4()
            # Criando um input zerado para os outros campos V (que são 28 ao todo)
            input_data = np.zeros((1, len(cols)))
            input_df = pd.DataFrame(input_data, columns=cols)
            input_df['V1'] = v1
            input_df['V4'] = v4
            input_df['Amount'] = amount
            
            prediction = model_fraude.predict(input_df)
            prob = model_fraude.predict_proba(input_df)[0][1]
            
            if prediction[0] == 1:
                st.error(f"🚨 **ALERTA DE FRAUDE:** {prob:.2%} de probabilidade.")
                st.write("**Ação sugerida:** Bloqueio temporário e envio de SMS de confirmação.")
            else:
                st.success(f"✅ **TRANSAÇÃO LEGÍTIMA:** {prob:.2%} de risco.")

    with col2:
        st.subheader("Análise Operacional")
        st.write("**Matriz de Confusão (Conceito):**")
        
        # Exemplo visual de Matriz de Confusão
        import plotly.figure_factory as ff
        z = [[480, 12], [8, 484]] # Exemplo baseado no seu relatório
        x = ['Legítima', 'Fraude']
        y = ['Legítima', 'Fraude']
        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
        st.plotly_chart(fig)
        
        st.info("""
        **Por que o Recall é a prioridade?**
        No setor financeiro, o custo de uma fraude não detectada ($122 em média) é muito superior ao custo de enviar uma notificação para um cliente legítimo.
        """)
