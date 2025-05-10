from fastapi import FastAPI, HTTPException
from app.schemas import InputData
import joblib
import pandas as pd
import os

app = FastAPI(title="API de Previsão de Pagamentos")

# Caminhos para os arquivos do modelo e dos codificadores
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_lightgbm.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "encoders.pkl")

# Carregando o modelo e os codificadores
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar os arquivos: {e}")

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convertendo os dados de entrada para dicionário
        input_dict = data.dict()

        # Aplicando os encoders para colunas categóricas
        for col in ['acao_cobranca', 'dia_semana_cobranca']:
            if col in encoders:
                le = encoders[col]
                if input_dict[col] in le.classes_:
                    input_dict[col] = int(le.transform([input_dict[col]])[0])
                else:
                    raise ValueError(f"Valor desconhecido para {col}: {input_dict[col]}")
            else:
                raise ValueError(f"Codificador não encontrado para {col}")

        # Ordem esperada das features
        feature_order = [
            'valor_cobrado', 'lead_lag_vencimento', 'numero_cobrancas',
            'parcelas_em_atraso_acumuladas', 'dias_desde_ultima_cobranca',
            'tamanho_ies', 'qtd_cursos_na_ies', 'alunos_por_curso_ies',
            'dia_semana_cobranca', 'semana_do_mes', 'acao_cobranca'
        ]

        # Criar o DataFrame com as colunas na ordem correta
        df_input = pd.DataFrame([[input_dict[feat] for feat in feature_order]], columns=feature_order)

        # Realizando a predição
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return {
            "prediction": int(prediction),
            "probabilidade_pagamento": round(float(probability), 4)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a previsão: {e}")
