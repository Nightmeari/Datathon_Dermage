# =============================================================================
# 4. RELAÇÃO ENTRE CLUSTER E PRODUTOS
# =============================================================================

import pandas as pd

print("=" * 60)
print(" ANÁLISE: PRODUTOS POR CLUSTER")
print("=" * 60)

# === CAMINHOS ===
PATH_FINAL = "df_final.csv"
PATH_CLIENTES = "df_clientes.csv"

# === LEITURA ===
df_final = pd.read_csv(PATH_FINAL)
df_clientes = pd.read_csv(PATH_CLIENTES)

print("\nColunas df_final:")
print(df_final.columns)

print("\nColunas df_clientes:")
print(df_clientes.columns)

# === MERGE ===
# ajustar nome da coluna se necessário (cliente_id / id_cliente)
COL_CLIENTE = "cli_document"

df_merged = df_final.merge(df_clientes, on=COL_CLIENTE)

print(f"\nDataset combinado: {df_merged.shape[0]:,} linhas")

df_merged = df_final.merge(df_clientes, on="cli_document")

# criar coluna categoria
df_merged["categoria"] = df_merged["categorias_pedido"].str.split("|")
df_merged = df_merged.explode("categoria")


if "cluster_kmeans" not in df_merged.columns:
    raise ValueError("Coluna 'cluster_kmeans' não encontrada!")

# =============================================================================
# 1. CONTAGEM DE CLIENTES POR CATEGORIA EM CADA CLUSTER
# =============================================================================

print("\nTop categorias por cluster (número de clientes únicos):\n")

df_cat_cluster = (
    df_merged
    .groupby(["cluster_kmeans", "categoria"])["cli_document"]
    .nunique()
    .reset_index(name="n_clientes")
)

# ordenar
df_cat_cluster = df_cat_cluster.sort_values(
    ["cluster_kmeans", "n_clientes"],
    ascending=[True, False]
)

# mostrar top 5 de cada cluster
for cluster in df_cat_cluster["cluster_kmeans"].unique():
    print(f"\n--- Cluster {cluster} ---")
    print(df_cat_cluster[df_cat_cluster["cluster_kmeans"] == cluster].head(5))


# =============================================================================
# 2. PARTICIPAÇÃO (%) DENTRO DO CLUSTER
# =============================================================================

print("\nDistribuição percentual dentro de cada cluster:\n")

df_pct = df_cat_cluster.copy()

df_pct["total_cluster"] = df_pct.groupby("cluster_kmeans")["n_clientes"].transform("sum")
df_pct["pct"] = (df_pct["n_clientes"] / df_pct["total_cluster"]) * 100

df_pct = df_pct.sort_values(["cluster_kmeans", "pct"], ascending=[True, False])

for cluster in df_pct["cluster_kmeans"].unique():
    print(f"\n--- Cluster {cluster} ---")
    print(df_pct[df_pct["cluster_kmeans"] == cluster][["categoria", "pct"]].head(5))


# =============================================================================
# 3. TAXA DE RECOMPRA POR CATEGORIA (CROSS COM CLUSTER)
# =============================================================================

print("\nTaxa de recompra por categoria e cluster:\n")

df_recompra = (
    df_merged
    .groupby(["cluster_kmeans", "categoria"])["recomprou"]
    .mean()
    .reset_index()
)

df_recompra["recompra_pct"] = df_recompra["recomprou"] * 100

df_recompra = df_recompra.sort_values(
    ["cluster_kmeans", "recompra_pct"],
    ascending=[True, False]
)

for cluster in df_recompra["cluster_kmeans"].unique():
    print(f"\n--- Cluster {cluster} ---")
    print(df_recompra[df_recompra["cluster_kmeans"] == cluster][["categoria", "recompra_pct"]].head(5))


print("\n" + "=" * 60)
print(" ANÁLISE CONCLUÍDA")
print("=" * 60)