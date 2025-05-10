from pydantic import BaseModel

class InputData(BaseModel):
    valor_cobrado: float
    lead_lag_vencimento: int
    numero_cobrancas: int
    parcelas_em_atraso_acumuladas: int
    dias_desde_ultima_cobranca: int
    tamanho_ies: int
    qtd_cursos_na_ies: int
    alunos_por_curso_ies: float
    mais_de_um_curso: int
    dia_semana_cobranca: int
    semana_do_mes: int
    acao_cobranca: str
    status_pagamento: str
