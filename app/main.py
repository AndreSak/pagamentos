from fastapi import FastAPI, HTTPException
from app.schemas import InputData
import joblib
import numpy as np
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
        # Convertendo os dados de entrada para um dicionário
        input_dict = data.dict()

        # Aplicando os codificadores às variáveis categóricas
        for col in ['acao_cobranca', 'status_pagamento']:
            if col in encoders:
                le = encoders[col]
                if input_dict[col] in le.classes_:
                    input_dict[col] = int(le.transform([input_dict[col]])[0])
                else:
                    raise ValueError(f"Valor desconhecido para {col}: {input_dict[col]}")
            else:
                raise ValueError(f"Codificador não encontrado para {col}")

        # Criando o array de entrada para o modelo
        input_array = np.array([[
            input_dict['valor_cobrado'],
            input_dict['lead_lag_vencimento'],
            input_dict['numero_cobrancas'],
            input_dict['parcelas_em_atraso_acumuladas'],
            input_dict['dias_desde_ultima_cobranca'],
            input_dict['tamanho_ies'],
            input_dict['qtd_cursos_na_ies'],
            input_dict['alunos_por_curso_ies'],
            input_dict['mais_de_um_curso'],
            input_dict['dia_semana_cobranca'],
            input_dict['semana_do_mes'],
            input_dict['acao_cobranca'],
            input_dict['status_pagamento']
        ]])

        # Realizando a previsão
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)[0][1]

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a previsão: {e}")
