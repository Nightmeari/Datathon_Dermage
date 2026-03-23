"""
=============================================================================
ANÁLISE EXPLORATÓRIA DE RECOMPRA — E-COMMERCE
=============================================================================
Inputs  : df_final    → nível de pedido (gerado pelo script analise_recompra.py)
          df_clientes → nível de cliente (RFM + flag de recompra)

Seções  :
  1. Configuração e helpers
  2. Taxa de recompra
  3. Distribuição do tempo até a segunda compra
  4. Janela crítica de recompra
  5. Perfil: recompradores vs não-recompradores
  6. Produto de entrada (categoria da 1ª compra)
  7. Categoria vs taxa de recompra
  8. Canal vs taxa de recompra
  9. Painel de insights automáticos
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURAÇÃO GERAL E HELPERS
# =============================================================================

# ── Paleta e estilo global ────────────────────────────────────────────────────
COR_PRINCIPAL  = "#2E86AB"   # azul petróleo
COR_DESTAQUE   = "#E84855"   # vermelho para contraste
COR_NEUTRO     = "#A8DADC"   # azul claro
COR_TEXTO      = "#1D1D1B"
PALETTE_2      = [COR_PRINCIPAL, COR_DESTAQUE]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi"       : 130,
    "axes.titleweight" : "bold",
    "axes.titlesize"   : 12,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ── Separador visual para prints ─────────────────────────────────────────────
SEP = "─" * 62

def titulo(texto):
    print(f"\n{'═' * 62}")
    print(f"  {texto}")
    print(f"{'═' * 62}")

def subtitulo(texto):
    print(f"\n{SEP}")
    print(f"  {texto}")
    print(SEP)

# ── Coluna de rótulo legível para recomprou ──────────────────────────────────
df_clientes["perfil"] = df_clientes["recomprou"].map({1: "Recomprou", 0: "Não recomprou"})

# =============================================================================
# 2. TAXA DE RECOMPRA GERAL
# =============================================================================

titulo("2. TAXA DE RECOMPRA GERAL")

total          = len(df_clientes)
recompraram    = df_clientes["recomprou"].sum()
nao_recompraram = total - recompraram
taxa_recompra  = recompraram / total * 100

print(f"  Total de clientes únicos   : {total:>8,}")
print(f"  Clientes que recompraram   : {recompraram:>8,}  ({taxa_recompra:.1f}%)")
print(f"  Clientes sem recompra      : {nao_recompraram:>8,}  ({100 - taxa_recompra:.1f}%)")

# ── Gráfico de pizza ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    [recompraram, nao_recompraram],
    labels=[f"Recomprou\n{taxa_recompra:.1f}%", f"Não recomprou\n{100 - taxa_recompra:.1f}%"],
    colors=PALETTE_2,
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=11),
)
ax.set_title("Taxa de Recompra — Base de Clientes", pad=14)
plt.tight_layout()
plt.savefig("fig_01_taxa_recompra.png", bbox_inches="tight")
plt.show()

# =============================================================================
# 3. DISTRIBUIÇÃO DO TEMPO ATÉ A SEGUNDA COMPRA
# =============================================================================

titulo("3. DISTRIBUIÇÃO DO TEMPO ATÉ A SEGUNDA COMPRA")

tempo = df_clientes["tempo_ate_segunda_compra"].dropna()

media_tempo   = tempo.mean()
mediana_tempo = tempo.median()
p75_tempo     = tempo.quantile(0.75)
p90_tempo     = tempo.quantile(0.90)

print(f"  N (clientes com 2ª compra) : {len(tempo):>8,}")
print(f"  Média                      : {media_tempo:>8.1f} dias")
print(f"  Mediana                    : {mediana_tempo:>8.1f} dias")
print(f"  Percentil 75               : {p75_tempo:>8.1f} dias")
print(f"  Percentil 90               : {p90_tempo:>8.1f} dias")

# ── Histograma ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(
    tempo.clip(upper=tempo.quantile(0.95)),   # trunca outliers extremos
    bins=40,
    color=COR_PRINCIPAL,
    edgecolor="white",
    linewidth=0.5,
    alpha=0.85,
)
ax.axvline(media_tempo,   color=COR_DESTAQUE, linestyle="--", linewidth=1.8,
           label=f"Média: {media_tempo:.0f}d")
ax.axvline(mediana_tempo, color="#F4A261",    linestyle=":",  linewidth=1.8,
           label=f"Mediana: {mediana_tempo:.0f}d")
ax.set_xlabel("Dias até a 2ª compra")
ax.set_ylabel("Número de clientes")
ax.set_title("Distribuição do Tempo até a Segunda Compra")
ax.legend()
plt.tight_layout()
plt.savefig("fig_02_histograma_tempo.png", bbox_inches="tight")
plt.show()

# =============================================================================
# 4. JANELA CRÍTICA DE RECOMPRA
# =============================================================================

titulo("4. JANELA CRÍTICA DE RECOMPRA")

# Definir faixas de tempo (dias)
bins   = [0, 7, 15, 30, 60, 90, 180, np.inf]
labels = ["0–7d", "7–15d", "15–30d", "30–60d", "60–90d", "90–180d", "180d+"]

df_clientes["faixa_recompra"] = pd.cut(
    df_clientes["tempo_ate_segunda_compra"],
    bins=bins,
    labels=labels,
    right=False,
)

tab_faixas = (
    df_clientes["faixa_recompra"]
    .value_counts()
    .reindex(labels)         # manter ordem correta
    .reset_index()
    .rename(columns={"index": "faixa", "faixa_recompra": "clientes"})
)
tab_faixas.columns = ["faixa", "clientes"]
tab_faixas["pct_recompradores"] = (tab_faixas["clientes"] / recompraram * 100).round(1)
tab_faixas["pct_acumulado"]     = tab_faixas["pct_recompradores"].cumsum().round(1)

print("\n  Distribuição de recompras por janela de tempo:")
print(tab_faixas.to_string(index=False))

# Janela com mais recompras
faixa_pico = tab_faixas.loc[tab_faixas["clientes"].idxmax(), "faixa"]
pct_30d    = tab_faixas[tab_faixas["faixa"].isin(["0–7d","7–15d","15–30d"])]["pct_recompradores"].sum()

# ── Barplot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
cores   = [COR_DESTAQUE if f == faixa_pico else COR_PRINCIPAL for f in tab_faixas["faixa"]]

bars = ax.bar(tab_faixas["faixa"], tab_faixas["clientes"], color=cores, edgecolor="white")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
ax.set_xlabel("Janela de tempo (dias)")
ax.set_ylabel("Número de clientes")
ax.set_title("Recompras por Janela de Tempo\n(barra em destaque = janela com mais recompras)")
plt.tight_layout()
plt.savefig("fig_03_janela_critica.png", bbox_inches="tight")
plt.show()

# =============================================================================
# 5. COMPARAÇÃO: RECOMPRADORES vs NÃO-RECOMPRADORES
# =============================================================================

titulo("5. PERFIL: RECOMPRADORES vs NÃO-RECOMPRADORES")

metricas   = ["monetario", "recencia", "frequencia"]
rotulos    = {"monetario": "Valor Total (R$)", "recencia": "Recência (dias)",
              "frequencia": "Frequência (pedidos)"}

tab_perfil = (
    df_clientes.groupby("perfil")[metricas]
    .agg(["mean", "median"])
    .round(2)
)
# Flatten multi-index de colunas
tab_perfil.columns = [f"{m}_{stat}" for m, stat in tab_perfil.columns]
print("\n  Médias e medianas por perfil:")
print(tab_perfil.to_string())

# ── Box plots lado a lado ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Distribuição de Métricas: Recomprou vs Não Recomprou", fontsize=13, fontweight="bold")

for ax, metrica in zip(axes, metricas):
    dados = df_clientes[[metrica, "perfil"]].dropna()
    # Truncar outliers para melhor visualização (percentil 95)
    cap = dados[metrica].quantile(0.95)
    dados = dados[dados[metrica] <= cap]

    sns.boxplot(
        data=dados,
        x="perfil",
        y=metrica,
        palette=PALETTE_2,
        width=0.45,
        linewidth=1.2,
        ax=ax,
    )
    ax.set_title(rotulos[metrica])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)

plt.tight_layout()
plt.savefig("fig_04_perfil_recompra.png", bbox_inches="tight")
plt.show()

# =============================================================================
# 6. PRODUTO DE ENTRADA (CATEGORIA DA 1ª COMPRA)
# =============================================================================

titulo("6. CATEGORIA DO PRODUTO DE ENTRADA (1ª COMPRA)")

# Filtrar somente primeiras compras
df_primeira = df_final[df_final["n_compra"] == 1][["cli_document", "categorias_pedido"]].copy()

# Explodir string separada por "|" → uma linha por categoria
df_primeira["categorias_pedido"] = df_primeira["categorias_pedido"].fillna("DESCONHECIDO")
df_explodido = (
    df_primeira
    .assign(categoria=df_primeira["categorias_pedido"].str.split("|"))
    .explode("categoria")
    .assign(categoria=lambda d: d["categoria"].str.strip().str.upper())
)

top_entrada = (
    df_explodido["categoria"]
    .value_counts()
    .head(10)
    .reset_index()
    .rename(columns={"index": "categoria", "categoria": "qtd"})
)
top_entrada.columns = ["categoria", "qtd"]

print("\n  Top 10 categorias na primeira compra:")
print(top_entrada.to_string(index=False))

# ── Barplot horizontal ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=top_entrada.sort_values("qtd"),
    x="qtd",
    y="categoria",
    palette=sns.light_palette(COR_PRINCIPAL, n_colors=10, reverse=True),
    ax=ax,
)
ax.set_title("Top 10 Categorias na Primeira Compra")
ax.set_xlabel("Número de clientes")
ax.set_ylabel("")
ax.bar_label(ax.containers[0], fmt="%d", padding=3, fontsize=9)
plt.tight_layout()
plt.savefig("fig_05_produto_entrada.png", bbox_inches="tight")
plt.show()

# =============================================================================
# 7. CATEGORIA vs TAXA DE RECOMPRA
# =============================================================================

titulo("7. CATEGORIA vs TAXA DE RECOMPRA")

# Associar cada cliente (primeira compra) à flag recomprou
df_cat_recompra = df_explodido.merge(
    df_clientes[["cli_document", "recomprou"]],
    on="cli_document",
    how="left",
).drop_duplicates(subset=["cli_document", "categoria"])

# Taxa de recompra por categoria (mín. 30 clientes para relevância estatística)
MIN_CLIENTES = 30

taxa_por_cat = (
    df_cat_recompra.groupby("categoria")
    .agg(total=("recomprou", "count"), recompraram=("recomprou", "sum"))
    .query(f"total >= {MIN_CLIENTES}")
    .assign(taxa_recompra=lambda d: d["recompraram"] / d["total"] * 100)
    .sort_values("taxa_recompra", ascending=False)
    .reset_index()
)

top5_alta  = taxa_por_cat.head(5)
top5_baixa = taxa_por_cat.tail(5)

print(f"\n  Categorias com MAIOR taxa de recompra (mín. {MIN_CLIENTES} clientes):")
print(top5_alta[["categoria", "total", "recompraram", "taxa_recompra"]].to_string(index=False))

print(f"\n  Categorias com MENOR taxa de recompra:")
print(top5_baixa[["categoria", "total", "recompraram", "taxa_recompra"]].to_string(index=False))

# ── Barplot das top + bottom 5 ────────────────────────────────────────────────
top5_concat = pd.concat([
    top5_alta.assign(grupo="Top 5 — maior recompra"),
    top5_baixa.assign(grupo="Top 5 — menor recompra"),
])

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=top5_concat,
    x="taxa_recompra",
    y="categoria",
    hue="grupo",
    palette=PALETTE_2,
    dodge=False,
    ax=ax,
)
ax.axvline(taxa_recompra, color="gray", linestyle="--", linewidth=1.3,
           label=f"Média geral: {taxa_recompra:.1f}%")
ax.set_title("Taxa de Recompra por Categoria (1ª compra)")
ax.set_xlabel("Taxa de recompra (%)")
ax.set_ylabel("")
ax.legend(title="", fontsize=9)
plt.tight_layout()
plt.savefig("fig_06_categoria_recompra.png", bbox_inches="tight")
plt.show()

# =============================================================================
# 8. CANAL vs TAXA DE RECOMPRA
# =============================================================================

titulo("8. CANAL vs TAXA DE RECOMPRA")

# Garantir 1 linha por cliente (usar a 1ª compra como canal de aquisição)
df_canal = (
    df_final[df_final["n_compra"] == 1][["cli_document", "canal"]]
    .drop_duplicates(subset="cli_document")
    .merge(df_clientes[["cli_document", "recomprou"]], on="cli_document", how="left")
)

df_canal["canal"] = df_canal["canal"].fillna("Desconhecido").str.strip().str.title()

taxa_canal = (
    df_canal.groupby("canal")
    .agg(total=("recomprou", "count"), recompraram=("recomprou", "sum"))
    .assign(taxa_recompra=lambda d: d["recompraram"] / d["total"] * 100)
    .reset_index()
    .sort_values("taxa_recompra", ascending=False)
)

print("\n  Taxa de recompra por canal de aquisição:")
print(taxa_canal.to_string(index=False))

# ── Barplot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
cores_canal = sns.color_palette(
    [COR_PRINCIPAL, COR_DESTAQUE, COR_NEUTRO, "#F4A261"], n_colors=len(taxa_canal)
)
bars = ax.bar(taxa_canal["canal"], taxa_canal["taxa_recompra"],
              color=cores_canal, edgecolor="white")
ax.axhline(taxa_recompra, color="gray", linestyle="--", linewidth=1.3,
           label=f"Média geral: {taxa_recompra:.1f}%")
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
ax.set_title("Taxa de Recompra por Canal de Aquisição")
ax.set_xlabel("")
ax.set_ylabel("Taxa de recompra (%)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig("fig_07_canal_recompra.png", bbox_inches="tight")
plt.show()

# =============================================================================
# 9. PAINEL DE INSIGHTS AUTOMÁTICOS
# =============================================================================

titulo("9. PAINEL DE INSIGHTS AUTOMÁTICOS")

# ── Helpers de cálculo ────────────────────────────────────────────────────────
melhor_canal = taxa_canal.iloc[0]["canal"]
taxa_melhor_canal = taxa_canal.iloc[0]["taxa_recompra"]

pior_canal = taxa_canal.iloc[-1]["canal"]
taxa_pior_canal = taxa_canal.iloc[-1]["taxa_recompra"]

cat_top1    = top5_alta.iloc[0]["categoria"]
taxa_cat_t1 = top5_alta.iloc[0]["taxa_recompra"]

cat_bot1    = top5_baixa.iloc[-1]["categoria"]
taxa_cat_b1 = top5_baixa.iloc[-1]["taxa_recompra"]

pct_90d = tab_faixas[tab_faixas["faixa"].isin(
    ["0–7d","7–15d","15–30d","30–60d","60–90d"]
)]["pct_recompradores"].sum()

media_monetario_rec    = df_clientes[df_clientes["recomprou"]==1]["monetario"].mean()
media_monetario_norec  = df_clientes[df_clientes["recomprou"]==0]["monetario"].mean()
mult_monetario         = media_monetario_rec / media_monetario_norec if media_monetario_norec > 0 else np.nan

# ── Impressão dos insights ────────────────────────────────────────────────────
insights = [
    f"📌  {taxa_recompra:.1f}% dos clientes realizaram ao menos uma segunda compra.",

    f"⏱   A janela mais crítica de recompra é '{faixa_pico}': "
    f"mais clientes retornam nessa faixa de tempo.",

    f"⏱   {pct_30d:.0f}% das recompras ocorrem nos primeiros 30 dias após a 1ª compra.",

    f"⏱   Aproximadamente {pct_90d:.0f}% das recompras ocorrem dentro de 90 dias. "
    f"Ações de reativação após esse período tendem a ser menos eficazes.",

    f"💰  Clientes que recompraram gastam em média "
    f"R$ {media_monetario_rec:,.2f} vs R$ {media_monetario_norec:,.2f} "
    f"({mult_monetario:.1f}x mais) dos que não recompraram.",

    f"🛍   A categoria com maior taxa de recompra é '{cat_top1}' "
    f"({taxa_cat_t1:.1f}%). Clientes que compram este produto têm maior chance de retornar.",

    f"🛍   A categoria com menor taxa de recompra é '{cat_bot1}' "
    f"({taxa_cat_b1:.1f}%). Pode exigir ações específicas de retenção.",

    f"📡  O canal com maior taxa de recompra é '{melhor_canal}' "
    f"({taxa_melhor_canal:.1f}%), sugerindo que clientes adquiridos por esse canal "
    f"têm maior fidelização.",

    f"📡  O canal com menor taxa de recompra é '{pior_canal}' "
    f"({taxa_pior_canal:.1f}%). Avaliar estratégias de nutrição pós-venda para esse canal.",

    f"🎯  Recomendação: focar campanhas de reativação na janela de '{faixa_pico}', "
    f"priorizando clientes adquiridos via '{pior_canal}' "
    f"e na categoria '{cat_bot1}' para elevar a taxa de recompra.",
]

for i, insight in enumerate(insights, 1):
    print(f"\n  [{i:02d}] {insight}")

print(f"\n{'═' * 62}")
print("  Análise exploratória concluída.")
print(f"  Gráficos salvos: fig_01 a fig_07")
print(f"{'═' * 62}\n")