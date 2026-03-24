import pandas as pd
import matplotlib.pyplot as plt

PATH_CLIENTES = "df_clientes.csv"

# === LEITURA ===
df_clientes = pd.read_csv(PATH_CLIENTES)

# filtrar só quem recomprou
df = df_clientes[df_clientes["recomprou"] == 1].copy()

# ordenar por tempo
df = df.sort_values("tempo_ate_segunda_compra")

# criar % acumulada
df["pct_acumulado"] = (df.reset_index().index + 1) / len(df) * 100

# plot
plt.figure()
plt.plot(df["tempo_ate_segunda_compra"], df["pct_acumulado"])

plt.xlabel("Dias até segunda compra")
plt.ylabel("% acumulado de clientes")
plt.title("Curva de recompra acumulada")


# filtrar só quem recomprou
df = df_clientes[df_clientes["recomprou"] == 1].copy()

# função pra calcular %
def pct_ate(dias):
    return (df["tempo_ate_segunda_compra"] <= dias).mean() * 100

# pontos importantes
marcos = [30, 60, 90, 117, 150, 180, 240, 300, 365]

for d in marcos:
    print(f"Até {d} dias: {pct_ate(d):.1f}%")