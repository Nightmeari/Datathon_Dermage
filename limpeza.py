"""
=============================================================================
ANÁLISE DE RECOMPRA DE CLIENTES — E-COMMERCE
=============================================================================
Objetivo : Preparar uma base de dados no nível de cliente com features
           para modelagem e análise de recompra (RFM + jornada de compras).

Inputs   : base_pedidos.csv | base_produtos.csv | familias.csv
Outputs  : df_final      → pedidos limpos e enriquecidos (1 linha/pedido)
           df_clientes   → agregado por cliente (RFM + recompra)
=============================================================================
"""

import pandas as pd
import numpy as np

# =============================================================================
# 0. CONFIGURAÇÕES GERAIS
# =============================================================================

# Caminhos dos arquivos de entrada
PATH_PEDIDOS  = "base_pedidos.csv"
PATH_PRODUTOS = "base_produtos.csv"
PATH_FAMILIAS = "familias.csv"

# Separador dos CSVs (ajuste se necessário: ";" ou ",")
SEP = ","

# =============================================================================
# 1. LEITURA DOS ARQUIVOS
# =============================================================================

print("=" * 60)
print("ETAPA 1 — Leitura dos arquivos")
print("=" * 60)

df_pedidos  = pd.read_csv(PATH_PEDIDOS,  sep=SEP, dtype=str)
df_produtos = pd.read_csv(PATH_PRODUTOS, sep=SEP, dtype=str)
df_familias = pd.read_csv(PATH_FAMILIAS, sep=SEP, dtype=str)

print(f"  base_pedidos  → {df_pedidos.shape[0]:>7,} linhas | {df_pedidos.shape[1]} colunas")
print(f"  base_produtos → {df_produtos.shape[0]:>7,} linhas | {df_produtos.shape[1]} colunas")
print(f"  familias      → {df_familias.shape[0]:>7,} linhas | {df_familias.shape[1]} colunas")

# =============================================================================
# 2. LIMPEZA E TIPAGEM — BASE DE PEDIDOS
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 2 — Limpeza: base_pedidos")
print("=" * 60)

# Normalizar nomes de colunas (lowercase + sem espaços extras)
df_pedidos.columns = df_pedidos.columns.str.strip().str.lower()

# Converter valor para numérico (vírgula como decimal → ponto)
df_pedidos["value"] = (
    df_pedidos["value"]
    .str.replace(",", ".", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

# Converter data para datetime
df_pedidos["data"] = pd.to_datetime(df_pedidos["data"], dayfirst=True, errors="coerce")

# Filtrar apenas pedidos faturados (status == 'invoiced')
antes = len(df_pedidos)
df_pedidos = df_pedidos[df_pedidos["status"].str.lower() == "invoiced"].copy()
depois = len(df_pedidos)
print(f"  Pedidos antes do filtro  : {antes:>7,}")
print(f"  Pedidos após filtro      : {depois:>7,}")
print(f"  Pedidos removidos        : {antes - depois:>7,}")

# Remover duplicatas de orderid (mantém a primeira ocorrência)
dup_pedidos = df_pedidos.duplicated(subset="orderid").sum()
if dup_pedidos > 0:
    print(f"  ⚠  Duplicatas em orderid removidas: {dup_pedidos}")
    df_pedidos = df_pedidos.drop_duplicates(subset="orderid", keep="first")

# =============================================================================
# 3. LIMPEZA E TIPAGEM — BASE DE PRODUTOS
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 3 — Limpeza: base_produtos")
print("=" * 60)

# Normalizar nomes de colunas
df_produtos.columns = df_produtos.columns.str.strip()

# Renomear colunas para facilitar o uso
df_produtos = df_produtos.rename(columns={
    "Order"               : "orderid",
    "Creation Date"       : "data_produto",
    "Client Document"     : "cli_document",
    "UF"                  : "uf",
    "Status"              : "status_produto",
    "Origin"              : "canal",
    "Payment System Name" : "forma_pagamento",
    "SKU Name"            : "sku_name",
})

# Converter data para datetime
df_produtos["data_produto"] = pd.to_datetime(
    df_produtos["data_produto"], dayfirst=True, errors="coerce"
)

# Filtrar apenas itens faturados
antes = len(df_produtos)
df_produtos = df_produtos[df_produtos["status_produto"].str.lower() == "faturado"].copy()
depois = len(df_produtos)
print(f"  Itens antes do filtro    : {antes:>7,}")
print(f"  Itens após filtro        : {depois:>7,}")
print(f"  Itens removidos          : {antes - depois:>7,}")

# =============================================================================
# 4. CRIAR CATEGORIA DO PRODUTO
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 4 — Criação da coluna 'categoria'")
print("=" * 60)

# A categoria é a PRIMEIRA PALAVRA do SKU Name (antes do primeiro espaço)
df_produtos["categoria"] = (
    df_produtos["sku_name"]
    .fillna("DESCONHECIDO")
    .str.strip()
    .str.split(" ")
    .str[0]
    .str.upper()
)

print(f"  Top 10 categorias mais frequentes:")
print(
    df_produtos["categoria"]
    .value_counts()
    .head(10)
    .rename_axis("categoria")
    .reset_index(name="qtd_itens")
    .to_string(index=False)
)

# =============================================================================
# 5. AGREGAR BASE DE PRODUTOS NO NÍVEL DE PEDIDO
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 5 — Agrupamento de produtos → nível de pedido")
print("=" * 60)

# ── 5a. Lista de categorias únicas por pedido (como string separada por |)
categorias_por_pedido = (
    df_produtos.groupby("orderid")["categoria"]
    .agg(lambda x: "|".join(sorted(set(x.dropna()))))
    .reset_index()
    .rename(columns={"categoria": "categorias_pedido"})
)

# ── 5b. Para canal, UF e forma de pagamento: pegar o valor mais frequente
#        (moda), útil quando um pedido tem itens com origens distintas.
def moda_segura(series):
    """Retorna a moda da série; se empate, retorna o primeiro valor."""
    moda = series.mode()
    return moda.iloc[0] if not moda.empty else np.nan

atributos_pedido = (
    df_produtos.groupby("orderid")[["canal", "uf", "forma_pagamento"]]
    .agg(moda_segura)
    .reset_index()
)

# ── 5c. Merge das duas agregações
df_produtos_pedido = categorias_por_pedido.merge(atributos_pedido, on="orderid", how="left")

print(f"  Pedidos únicos na base de produtos: {len(df_produtos_pedido):>7,}")

# =============================================================================
# 6. MERGE — PEDIDOS + PRODUTOS
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 6 — Merge base_pedidos ↔ base_produtos")
print("=" * 60)

df_final = df_pedidos.merge(
    df_produtos_pedido,
    left_on="orderid",
    right_on="orderid",
    how="left",          # mantém todos os pedidos faturados
    validate="1:1",      # garante que não há duplicatas de orderid
)

# Verificação rápida de cobertura do merge
nao_encontrados = df_final["categorias_pedido"].isna().sum()
print(f"  Total de pedidos no df_final         : {len(df_final):>7,}")
print(f"  Pedidos sem match em base_produtos   : {nao_encontrados:>7,}")

# =============================================================================
# 7. ORDENAR POR CLIENTE E DATA
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 7 — Ordenação por cliente e data")
print("=" * 60)

df_final = df_final.sort_values(
    by=["cli_document", "data"],
    ascending=[True, True],
    na_position="last",
).reset_index(drop=True)

print("  Dados ordenados por cli_document ↑ e data ↑")

# =============================================================================
# 8. VARIÁVEIS DE JORNADA — NÍVEL DE PEDIDO
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 8 — Variáveis de jornada por cliente")
print("=" * 60)

# ── 8a. Número sequencial da compra do cliente (1 = primeira compra)
df_final["n_compra"] = (
    df_final.groupby("cli_document").cumcount() + 1
)

# ── 8b. Tempo desde a última compra (dias) — NaN para a 1ª compra
df_final["dias_desde_ultima_compra"] = (
    df_final.groupby("cli_document")["data"].diff().dt.days
)

print("  Colunas criadas: n_compra | dias_desde_ultima_compra")
print(
    df_final[["n_compra", "dias_desde_ultima_compra"]]
    .describe()
    .round(1)
    .to_string()
)

# =============================================================================
# 9. BASE NO NÍVEL DE CLIENTE (RFM)
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 9 — Agregação por cliente (RFM)")
print("=" * 60)

# Data de referência = dia mais recente do dataset
data_referencia = df_final["data"].max()
print(f"  Data de referência (máx do dataset): {data_referencia.date()}")

df_clientes = (
    df_final.groupby("cli_document")
    .agg(
        frequencia    = ("orderid", "count"),
        monetario     = ("value",   "sum"),
        ultima_compra = ("data",    "max"),
        primeira_compra = ("data",  "min"),
        tipo_cliente  = ("tipo_cliente", "first"),   # NOVO / ANTIGO
        uf            = ("uf",      lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
    )
    .reset_index()
)

# ── Recência: dias desde a última compra até a data de referência
df_clientes["recencia"] = (
    data_referencia - df_clientes["ultima_compra"]
).dt.days

# ── Flag de recompra: 1 se o cliente comprou mais de uma vez
df_clientes["recomprou"] = (df_clientes["frequencia"] > 1).astype(int)

# =============================================================================
# 10. TEMPO ATÉ A SEGUNDA COMPRA
# =============================================================================

print("\n" + "=" * 60)
print("ETAPA 10 — Tempo até a segunda compra")
print("=" * 60)

# Selecionar apenas pedidos de n_compra == 2 para obter a data da 2ª compra
segunda_compra = (
    df_final[df_final["n_compra"] == 2][["cli_document", "data"]]
    .rename(columns={"data": "data_segunda_compra"})
)

# Merge com df_clientes para calcular o delta
df_clientes = df_clientes.merge(segunda_compra, on="cli_document", how="left")

df_clientes["tempo_ate_segunda_compra"] = (
    (df_clientes["data_segunda_compra"] - df_clientes["primeira_compra"]).dt.days
)

# Remover coluna auxiliar
df_clientes = df_clientes.drop(columns=["data_segunda_compra"])

print("  Coluna criada: tempo_ate_segunda_compra (dias)")

# =============================================================================
# 11. RESUMO EXECUTIVO
# =============================================================================

print("\n" + "=" * 60)
print("RESUMO EXECUTIVO")
print("=" * 60)

total_clientes    = len(df_clientes)
total_recompraram = df_clientes["recomprou"].sum()
pct_recompra      = total_recompraram / total_clientes * 100

media_tempo       = df_clientes["tempo_ate_segunda_compra"].mean()
mediana_tempo     = df_clientes["tempo_ate_segunda_compra"].median()

print(f"  Total de clientes          : {total_clientes:>8,}")
print(f"  Clientes que recompraram   : {total_recompraram:>8,}  ({pct_recompra:.1f}%)")
print(f"  Clientes que NÃO recompraram: {total_clientes - total_recompraram:>7,}  ({100 - pct_recompra:.1f}%)")
print()
print(f"  Tempo até 2ª compra — média  : {media_tempo:>6.1f} dias")
print(f"  Tempo até 2ª compra — mediana: {mediana_tempo:>6.1f} dias")
print()
print(f"  Frequência média de compras  : {df_clientes['frequencia'].mean():>6.2f}")
print(f"  Ticket médio por cliente     : R$ {df_clientes['monetario'].mean():>10,.2f}")
print(f"  Recência média (dias)        : {df_clientes['recencia'].mean():>6.1f}")

# =============================================================================
# 12. PREVIEW DAS BASES FINAIS
# =============================================================================

print("\n" + "=" * 60)
print("PREVIEW — df_final (pedidos enriquecidos)")
print("=" * 60)
print(df_final.head(5).to_string())

print("\n" + "=" * 60)
print("PREVIEW — df_clientes (base por cliente)")
print("=" * 60)
print(df_clientes.head(10).to_string())

print("\n" + "=" * 60)
print("COLUNAS FINAIS")
print("=" * 60)
print("\ndf_final:")
for col in df_final.columns:
    print(f"  • {col} ({df_final[col].dtype})")

print("\ndf_clientes:")
for col in df_clientes.columns:
    print(f"  • {col} ({df_clientes[col].dtype})")

# =============================================================================
# 13. EXPORTAÇÃO (opcional — descomente para salvar)
# =============================================================================

df_final.to_csv("df_final.csv", index=False, sep=",", encoding="utf-8-sig")
df_clientes.to_csv("df_clientes.csv", index=False, sep=",", encoding="utf-8-sig")
print("\n  ✅ Arquivos exportados com sucesso.")