import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar os dados
csv_filename = 'results_test_datadeval_osrm_2003.csv' 
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    # Caso o script rode em um diretório acima
    df = pd.read_csv('tests/results_test_datadeval_osrm_2003.csv')

# 2. Pré-processamento e Limpeza
if 'brkga_unique' in df['strategy'].values:
    df = df[df['strategy'] != 'brkga_unique'].copy()

# Mapeamento com as chaves exatas do CSV
name_map = {
    'first_fit+brkga': 'First-Fit + BRKGA',
    'ckmeans+brkga': 'CKMeans + BRKGA',
    'first_fit+cheapest_insertion': 'First-Fit + Cheapest Insertion',
    'ckmeans+cheapest_insertion': 'CKMeans + Cheapest Insertion',
    'greedy': 'Greedy',
    'manual_assignment': 'Manual Assignment'
}
df['strategy'] = df['strategy'].map(name_map).fillna(df['strategy'])

# 3. Configurações de estilo global
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = ':'

# Estilos ajustados para melhor legibilidade
markers = ['o', 's', '^', 'D', 'X', 'P']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 8))]
base_color = '#404040' # Cinza escuro elegante no lugar do preto absoluto

df_vehicles = df.groupby(['strategy', 'n_vehicles']).mean(numeric_only=True).reset_index()
strategies = df['strategy'].unique()

# ---------------------------------------------------------
# GRÁFICO 1: Penalidade Média vs Número de Veículos
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))
for i, strategy in enumerate(strategies):
    subset = df_vehicles[df_vehicles['strategy'] == strategy]
    plt.plot(subset['n_vehicles'], subset['avg_penalty'], 
             marker=markers[i], linestyle=linestyles[i], color=base_color, 
             linewidth=1.8, markersize=8, 
             markerfacecolor='white', markeredgewidth=1.5, # Marcador vazado
             label=strategy, zorder=3)

plt.xlabel('Número de Veículos', fontsize=12)
plt.ylabel('Penalidade Média', fontsize=12)
plt.xticks([1, 2, 3, 4, 5])
# Legenda com fundo branco suave para não confundir com as linhas
plt.legend(frameon=True, fontsize=10, loc='best', facecolor='white', edgecolor='#E0E0E0', framealpha=0.9)
plt.tight_layout()
plt.savefig('plot_penalidade_vs_veiculos.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# GRÁFICO 2: Tempo de Execução vs Número de Veículos
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))
for i, strategy in enumerate(strategies):
    subset = df_vehicles[df_vehicles['strategy'] == strategy]
    plt.plot(subset['n_vehicles'], subset['execution_time'], 
             marker=markers[i], linestyle=linestyles[i], color=base_color, 
             linewidth=1.8, markersize=8, 
             markerfacecolor='white', markeredgewidth=1.5,
             label=strategy, zorder=3)

plt.xlabel('Número de Veículos', fontsize=12)
plt.ylabel('Tempo Médio de Execução CPU (s)', fontsize=12)
plt.xticks([1, 2, 3, 4, 5])
plt.legend(frameon=True, fontsize=10, loc='best', facecolor='white', edgecolor='#E0E0E0', framealpha=0.9)
plt.tight_layout()
plt.savefig('plot_tempo_vs_veiculos.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# GRÁFICO 3: Trade-off (Penalidade Média vs Tempo Médio CPU)
# ---------------------------------------------------------
df_tradeoff = df.groupby('strategy')[['execution_time', 'avg_penalty']].mean().reset_index()

plt.figure(figsize=(10, 6))
for i, row in df_tradeoff.iterrows():
    idx = list(strategies).index(row['strategy'])
    
    # Pontos de dispersão aumentados e com fundo branco
    plt.scatter(row['execution_time'], row['avg_penalty'], 
                marker=markers[idx], color=base_color, s=120, 
                facecolors='white', edgecolors=base_color, linewidth=2, zorder=5)
    
    # Textos com caixa de fundo branca para fácil leitura mesmo sobrepostos
    plt.annotate(row['strategy'], 
                 (row['execution_time'], row['avg_penalty']),
                 xytext=(15, 0), textcoords='offset points', 
                 fontsize=11, family='serif', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", lw=1, alpha=0.9),
                 zorder=10)

plt.xlabel('Tempo Médio de Execução CPU (s)', fontsize=12)
plt.ylabel('Penalidade Média', fontsize=12)
plt.margins(x=0.25) # Garante que os nomes longos não cortem na direita
plt.tight_layout()
plt.savefig('plot_tradeoff_penalidade_tempo.png', dpi=300)
plt.close()
