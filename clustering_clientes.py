# =============================================================================
# SEGMENTAÇÃO DE CLIENTES - K-Means + DBSCAN
# Autor : Cientista de Dados
# Objetivo: Agrupar clientes por comportamento de compra e gerar insights
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#  Estilo global dos gráficos 
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

SEED = 42
np.random.seed(SEED)


# =============================================================================
# 0. CARREGAMENTO DOS DADOS REAIS
# =============================================================================

print("=" * 60)
print(" Carregando base de clientes")
print("=" * 60)

PATH_CLIENTES = "df_clientes.csv"

df_clientes = pd.read_csv(PATH_CLIENTES)

# Converter datas (se existirem)
if "ultima_compra" in df_clientes.columns:
    df_clientes["ultima_compra"] = pd.to_datetime(df_clientes["ultima_compra"], errors="coerce")

if "primeira_compra" in df_clientes.columns:
    df_clientes["primeira_compra"] = pd.to_datetime(df_clientes["primeira_compra"], errors="coerce")

# Garantir colunas obrigatórias
colunas_obrigatorias = ["recencia", "frequencia", "monetario"]

for col in colunas_obrigatorias:
    if col not in df_clientes.columns:
        raise ValueError(f"Coluna obrigatória ausente: {col}")

# Garantir coluna de tempo até segunda compra
if "tempo_ate_segunda_compra" not in df_clientes.columns:
    df_clientes["tempo_ate_segunda_compra"] = np.nan

# Converter para numérico
colunas_numericas = ["recencia", "frequencia", "monetario", "tempo_ate_segunda_compra", "recomprou"]

for col in colunas_numericas:
    if col in df_clientes.columns:
        df_clientes[col] = pd.to_numeric(df_clientes[col], errors="coerce")

df_clientes = df_clientes.reset_index(drop=True)

print(f" Dataset carregado: {df_clientes.shape[0]:,} clientes | {df_clientes.shape[1]} colunas\n")
print(df_clientes.describe().round(2))
print()

# =============================================================================
# 1. PREPARAÇÃO DOS DADOS
# =============================================================================
print("---- [1/9] Preparação dos dados ----------------------\n")

FEATURES = ["recencia", "frequencia", "monetario", "tempo_ate_segunda_compra"]

# 1a. Preenchimento de nulos: mediana de tempo_ate_segunda_compra
mediana_t2c = df_clientes["tempo_ate_segunda_compra"].median()
df_clientes["tempo_ate_segunda_compra"].fillna(mediana_t2c, inplace=True)
print(f"  - tempo_ate_segunda_compra -> nulos preenchidos com mediana = {mediana_t2c:.1f} dias")


# Preencher TODOS os nulos de uma vez (forma segura)
X_raw = df_clientes[FEATURES].copy()

print("\nNulos antes do tratamento:")
print(X_raw.isnull().sum())

# Preencher com mediana corretamente
X_raw = X_raw.fillna(X_raw.median())

# Garantir que não há mais nulos
print("\nNulos depois do tratamento:")
print(X_raw.isnull().sum())

# 1c. Escalonamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"  - Features escaladas com StandardScaler: {FEATURES}")
print(f"  - Shape final para clustering: {X_scaled.shape}\n")


# =============================================================================
# 2. SELEÇÃO DO MELHOR K - Elbow + Silhouette
# =============================================================================
print("---- [2/9] Seleção do melhor K (K-Means) --------------\n")

K_RANGE = range(2, 9)
inertias, silhouettes = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

#  Gráfico Elbow + Silhouette lado a lado 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Seleção do número ótimo de clusters (K)", fontweight="bold")

axes[0].plot(K_RANGE, inertias, "o-", color="#2a9d8f", lw=2.5, ms=8)
axes[0].set(title="Método do Cotovelo (Elbow)", xlabel="Número de clusters (K)", ylabel="Inércia")
axes[0].grid(True, alpha=0.4)

axes[1].plot(K_RANGE, silhouettes, "s-", color="#e76f51", lw=2.5, ms=8)
axes[1].set(title="Silhouette Score", xlabel="Número de clusters (K)", ylabel="Score")
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("elbow_silhouette.png", bbox_inches="tight")
plt.show()

#  Escolha automática: maior Silhouette 
BEST_K = list(K_RANGE)[np.argmax(silhouettes)]
print(f"  -> Melhor K escolhido automaticamente: K = {BEST_K}")
print(f"    (Silhouette Score = {max(silhouettes):.3f})\n")


# =============================================================================
# 3. TREINAR K-MEANS COM O MELHOR K
# =============================================================================
print("---- [3/9] Treinamento do K-Means --------------------------\n")

kmeans = KMeans(n_clusters=BEST_K, random_state=SEED, n_init=15)
df_clientes["cluster_kmeans"] = kmeans.fit_predict(X_scaled)
print(f"  - Clusters atribuídos: {sorted(df_clientes['cluster_kmeans'].unique())}")
print(f"  - Distribuição:\n{df_clientes['cluster_kmeans'].value_counts().sort_index()}\n")


# =============================================================================
# 4. ANÁLISE DOS CLUSTERS
# =============================================================================
print("---- [4/9] Análise descritiva dos clusters -----------------\n")

resumo = (
    df_clientes.groupby("cluster_kmeans")
    .agg(
        n_clientes        = ("cluster_kmeans",          "count"),
        recencia_media    = ("recencia",                "mean"),
        frequencia_media  = ("frequencia",              "mean"),
        monetario_medio   = ("monetario",               "mean"),
        t2c_media         = ("tempo_ate_segunda_compra","mean"),
        taxa_recompra     = ("recomprou",               "mean"),
    )
    .round(2)
)
print(resumo.to_string(), "\n")


# =============================================================================
# 5. NOMEAR CLUSTERS DINAMICAMENTE
# =============================================================================
print("---- [5/9] Nomeação automática dos clusters ------------------\n")

def nomear_cluster(row: pd.Series, medians: pd.Series) -> str:
    """
    Atribui um rótulo de negócio ao cluster com base nos valores
    relativos às medianas globais dos clusters.
    """
    alto_valor    = row["monetario_medio"]  > medians["monetario_medio"]
    alta_freq     = row["frequencia_media"] > medians["frequencia_media"]
    baixa_recencia= row["recencia_media"]   < medians["recencia_media"]   # comprou recentemente
    alta_recompra = row["taxa_recompra"]    > 0.70
    baixa_recompra= row["taxa_recompra"]    < 0.40

    if alto_valor and alta_freq and baixa_recencia:
        return " Clientes Fiéis de Alto Valor"
    elif alto_valor and not alta_freq:
        return " Clientes VIP Esporádicos"
    elif alta_freq and not alto_valor and alta_recompra:
        return " Clientes Frequentes de Baixo Ticket"
    elif baixa_recompra and not baixa_recencia:
        return "  Clientes em Risco de Churn"
    elif not alto_valor and not alta_freq and not baixa_recencia:
        return " Clientes Inativos"
    elif not alto_valor and baixa_recencia and not alta_recompra:
        return " Clientes Novos com Potencial"
    else:
        return " Clientes Regulares"

medians = resumo.median()
resumo["nome_cluster"] = resumo.apply(nomear_cluster, axis=1, medians=medians)

# Mapear nomes de volta ao dataframe principal
mapa_nomes = resumo["nome_cluster"].to_dict()
df_clientes["nome_cluster"] = df_clientes["cluster_kmeans"].map(mapa_nomes)

print("  Rótulos atribuídos:\n")
for idx, row in resumo.iterrows():
    print(f"  Cluster {idx} -> {row['nome_cluster']}")
print()


# =============================================================================
# 6. VISUALIZAÇÕES K-MEANS
# =============================================================================
print("---- [6/9] Gerando visualizações K-Means ------------------\n")

PALETTE = sns.color_palette("tab10", n_colors=BEST_K)

#  6a. Scatter: Frequência × Monetário 
fig, ax = plt.subplots(figsize=(9, 6))
for k, nome in mapa_nomes.items():
    mask = df_clientes["cluster_kmeans"] == k
    ax.scatter(
        df_clientes.loc[mask, "frequencia"],
        df_clientes.loc[mask, "monetario"],
        label=nome, alpha=0.65, s=45, color=PALETTE[k], edgecolors="white", lw=0.3,
    )
ax.set(title="Segmentação: Frequência x Valor Monetário",
       xlabel="Frequência (nº de pedidos)", ylabel="Monetário (R$)")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("scatter_freq_monetario.png", bbox_inches="tight")
plt.show()

# 6b. Boxplots das 4 features por cluster 
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Distribuição das variáveis por cluster", fontweight="bold", fontsize=14)

plot_cols = [
    ("recencia",                 "Recência (dias)"),
    ("frequencia",               "Frequência (pedidos)"),
    ("monetario",                "Monetário (R$)"),
    ("tempo_ate_segunda_compra", "Tempo até 2ª compra (dias)"),
]

for ax, (col, titulo) in zip(axes.flatten(), plot_cols):
    sns.boxplot(
        data=df_clientes, x="cluster_kmeans", y=col,
        hue="cluster_kmeans", palette=PALETTE, legend=False,
        ax=ax, flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    ax.set(title=titulo, xlabel="Cluster", ylabel="")
    ax.set_xticks(range(BEST_K))
    ax.set_xticklabels(
        [f"C{k}\n{mapa_nomes[k].split(' ', 1)[0]}" for k in range(BEST_K)],
        fontsize=7,
    )

plt.tight_layout()
plt.savefig("boxplots_clusters.png", bbox_inches="tight")
plt.show()

#  6c. Distribuição de tamanho dos clusters 
contagem = df_clientes["nome_cluster"].value_counts()
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(contagem.index, contagem.values, color=PALETTE[:len(contagem)])
ax.bar_label(bars, padding=4, fontsize=9)
ax.set(title="Número de clientes por segmento", xlabel="Quantidade", ylabel="")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("distribuicao_clusters.png", bbox_inches="tight")
plt.show()

#  6d. Heatmap de médias por cluster (normalizado) 
heat_data = resumo[["recencia_media","frequencia_media","monetario_medio",
                     "t2c_media","taxa_recompra"]].copy()
heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())
heat_norm.index = [mapa_nomes[i] for i in heat_norm.index]

fig, ax = plt.subplots(figsize=(10, max(3, BEST_K * 0.8)))
sns.heatmap(
    heat_norm, annot=heat_data.round(1), fmt="g",
    cmap="YlOrRd", linewidths=0.5, ax=ax,
    cbar_kws={"label": "Score normalizado"},
    xticklabels=["Recência","Frequência","Monetário","T. 2ª Compra","Taxa Recompra"],
)
ax.set(title="Perfil dos segmentos (escala normalizada)", ylabel="")
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("heatmap_clusters.png", bbox_inches="tight")
plt.show()

#  6e. PCA 2D - visão geral dos clusters 
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(8, 6))
for k, nome in mapa_nomes.items():
    mask = df_clientes["cluster_kmeans"] == k
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=nome, alpha=0.65, s=35, color=PALETTE[k], edgecolors="white", lw=0.3)
ax.set(title="Clusters - Projeção PCA 2D",
       xlabel=f"PC1 ({var_exp[0]:.1f}% variância)",
       ylabel=f"PC2 ({var_exp[1]:.1f}% variância)")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("pca_clusters.png", bbox_inches="tight")
plt.show()


# =============================================================================
# 7. DBSCAN - Detecção de outliers e clusters alternativos
# =============================================================================
print("----[7/9] DBSCAN -----------------------\n")

# Parâmetros: eps e min_samples ajustados com base no tamanho do dataset
# eps: raio da vizinhança; min_samples: mínimo de pontos para formar núcleo
EPS        = 0.55
MIN_SAMPLES = max(5, int(0.01 * len(df_clientes)))   # 1% do total, mínimo 5

dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
df_clientes["cluster_dbscan"] = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(df_clientes["cluster_dbscan"])) - (
    1 if -1 in df_clientes["cluster_dbscan"].values else 0
)
n_outliers = (df_clientes["cluster_dbscan"] == -1).sum()

print(f"  - eps={EPS} | min_samples={MIN_SAMPLES}")
print(f"  - Clusters DBSCAN encontrados: {n_clusters_db}")
print(f"  - Outliers (ruído, label=-1):  {n_outliers} clientes\n")


# =============================================================================
# 8. ANÁLISE DOS OUTLIERS (DBSCAN label = -1)
# =============================================================================
print("---- [8/9] Análise dos outliers (DBSCAN) ---------------------\n")

outliers = df_clientes[df_clientes["cluster_dbscan"] == -1].copy()

if outliers.empty:
    print("  Nenhum outlier detectado com os parâmetros atuais.\n")
else:
    print(f"  {len(outliers)} outliers detectados.\n")
    print("  Estatísticas dos outliers:\n")
    print(outliers[FEATURES + ["recomprou"]].describe().round(2).to_string(), "\n")

    # Classificação de sub-tipo de outlier
    limiar_vip       = df_clientes["monetario"].quantile(0.95)
    limiar_freq_alto = df_clientes["frequencia"].quantile(0.95)
    limiar_rec_alto  = df_clientes["recencia"].quantile(0.90)

    def tipo_outlier(row):
        if row["monetario"] >= limiar_vip:
            return " Cliente VIP (alto valor)"
        elif row["frequencia"] >= limiar_freq_alto:
            return " Comprador Compulsivo (alta frequência)"
        elif row["recencia"] >= limiar_rec_alto:
            return " Cliente Fantasma (sem comprar há muito tempo)"
        else:
            return " Comportamento Atípico"

    outliers["tipo_outlier"] = outliers.apply(tipo_outlier, axis=1)
    df_clientes.loc[outliers.index, "tipo_outlier"] = outliers["tipo_outlier"]

    print("  Distribuição dos tipos de outlier:")
    print(outliers["tipo_outlier"].value_counts().to_string(), "\n")

    # Scatter de outliers vs. clientes normais
    fig, ax = plt.subplots(figsize=(9, 6))
    normais = df_clientes[df_clientes["cluster_dbscan"] != -1]
    ax.scatter(normais["frequencia"], normais["monetario"],
               alpha=0.4, s=25, color="#adb5bd", label="Normais")
    for tipo, grp in outliers.groupby("tipo_outlier"):
        ax.scatter(grp["frequencia"], grp["monetario"],
                   s=90, label=tipo, edgecolors="black", lw=0.7, zorder=5)
    ax.set(title="DBSCAN - Outliers destacados",
           xlabel="Frequência", ylabel="Monetário (R$)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("dbscan_outliers.png", bbox_inches="tight")
    plt.show()


# =============================================================================
# 9. INSIGHTS AUTOMÁTICOS DE NEGÓCIO
# =============================================================================
print("---- [9/9] Insights automáticos de negócio ------------------\n")

def gerar_insight(row: pd.Series) -> str:
    nome  = row["nome_cluster"]
    linhas = [f" {nome}"]
    linhas.append(f"    {int(row['n_clientes'])} clientes ({row['n_clientes']/df_clientes.shape[0]*100:.1f}% da base)")
    linhas.append(f"    Ticket médio: R$ {row['monetario_medio']:,.2f} | Frequência: {row['frequencia_media']:.1f} pedidos")
    linhas.append(f"    Última compra: {row['recencia_media']:.0f} dias atrás | Taxa de recompra: {row['taxa_recompra']*100:.0f}%")

    # Recomendações contextuais
    if row["taxa_recompra"] >= 0.70 and row["monetario_medio"] > df_clientes["monetario"].median():
        linhas.append("    AÇÃO: Programa de fidelidade e upsell - são os clientes mais valiosos.")
    elif row["taxa_recompra"] < 0.40:
        linhas.append("    AÇÃO: Campanha de reativação urgente - alto risco de churn.")
    elif row["recencia_media"] < 30 and row["frequencia_media"] <= 2:
        linhas.append("    AÇÃO: Nutrição via e-mail/push - clientes novos com potencial de fidelização.")
    elif row["monetario_medio"] > df_clientes["monetario"].quantile(0.80) and row["frequencia_media"] < 3:
        linhas.append("    AÇÃO: Ofertas exclusivas e atendimento premium - compram pouco, mas gastam muito.")
    else:
        linhas.append("    AÇÃO: Manter engajamento com promoções segmentadas.")

    return "\n".join(linhas)

print("=" * 60)
print(" RELATÓRIO DE SEGMENTAÇÃO - INSIGHTS DE NEGÓCIO")
print("=" * 60)
for _, row in resumo.iterrows():
    print()
    print(gerar_insight(row))
print()

# Insights sobre outliers
if not outliers.empty:
    print("-" * 60)
    print(f"\n OUTLIERS (DBSCAN): {len(outliers)} clientes com comportamento atípico")
    print(f"   Monetário médio dos outliers: R$ {outliers['monetario'].mean():,.2f}")
    print(f"    vs. média geral:              R$ {df_clientes['monetario'].mean():,.2f}")
    if outliers["monetario"].mean() > df_clientes["monetario"].quantile(0.80):
        print("   Muitos outliers são clientes VIP - merecem atendimento personalizado!")
    else:
        print("    Outliers podem representar anomalias operacionais - investigar individualmente.")

# Resumo final da tabela
print("\n" + "=" * 60)
print(" TABELA RESUMO FINAL")
print("=" * 60)
print(resumo[[
    "nome_cluster","n_clientes","recencia_media",
    "frequencia_media","monetario_medio","taxa_recompra"
]].to_string())

print(f"\nColunas adicionadas ao df_clientes: cluster_kmeans - cluster_dbscan - nome_cluster")
print(f"Gráficos salvos: elbow_silhouette - scatter_freq_monetario - boxplots_clusters")
print(f"                    distribuicao_clusters - heatmap_clusters - pca_clusters - dbscan_outliers")