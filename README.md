# Sistema de Otimização de Rotas de Veículos

Este projeto implementa uma simulação para otimização de rotas de entrega em tempo real, utilizando diferentes estratégias algorítmicas para resolver o Problema de Roteamento de Veículos (VRP) com janelas de tempo.

## Como o Sistema é Executado

O sistema funciona como uma **simulação baseada em eventos** para otimização de rotas de entrega em tempo real. O fluxo principal, orquestrado pela classe `Core`, é o seguinte:

1.  **Inicialização:** A simulação é configurada com veículos, um depósito de origem e a escolha dos algoritmos a serem usados (`Config`).
2.  **Linha do Tempo de Eventos:** O sistema mantém uma fila de eventos priorizados por tempo (criação de pedidos, pedidos prontos, prazos, etc.). A simulação avança minuto a minuto, processando os eventos que já aconteceram.
3.  **Lógica de Decisão:** A cada minuto, o sistema verifica se há entregas prontas e veículos disponíveis. Se houver, a `routing_decision_logic` é acionada.
4.  **Aplicação do Algoritmo:** Neste ponto, a estratégia de algoritmo configurada é executada. Existem duas abordagens principais:
    *   **Duas Etapas (Cluster + Rota):** Primeiro, um algoritmo de **agrupamento (clustering)** atribui um conjunto de entregas a cada veículo. Depois, um algoritmo de **roteamento (routing)** define a melhor sequência de visitas para cada um desses veículos.
    *   **Etapa Única (Híbrido):** Um único algoritmo, mais complexo, resolve os dois problemas de uma vez: ele decide qual veículo atende quais entregas **e** a ordem das visitas, tudo simultaneamente.
5.  **Política de Despacho (ASAP vs. JIT):** Após gerar as rotas, o sistema decide quando despachá-las.
    *   **ASAP (As Soon As Possible):** Envia o veículo imediatamente. Usado em "modo de urgência" (pedidos com prazo curto ou muitos pedidos na fila).
    *   **JIT (Just-in-Time):** Se a rota tem folga de tempo, o sistema pode atrasar a partida de propósito. O objetivo é aumentar a chance de consolidar um novo pedido que possa chegar em breve na mesma rota, otimizando a eficiência.
6.  **Atualização de Estado:** O estado dos veículos (em rota) e das entregas (despachado) é atualizado, e novos eventos (como a chegada da entrega e o retorno do veículo) são agendados na fila.
7.  **Fim da Simulação:** O ciclo se repete até o fim do tempo de simulação, e um relatório de desempenho é gerado.

---

## Descrição dos Algoritmos

Os algoritmos são divididos em três categorias:

### 1. Algoritmos de Agrupamento (Clustering)

*O objetivo é atribuir entregas aos veículos.*

#### `CKMEANS` (Capacitated K-Means)
Um algoritmo de agrupamento inteligente. Ele forma clusters de entregas geograficamente próximas, mas com uma regra crucial: a soma dos "tamanhos" das entregas em um cluster não pode exceder a capacidade do veículo. Para isso, ele usa técnicas de otimização matemática (Programação Inteira Mista).

```plaintext
FUNÇÃO CKMeans(entregas, veículos):
  1. Inicia os centros dos clusters usando o K-Means padrão.
  2. PARA cada iteração de 1 a max_iters:
  3.   Calcula a matriz de distância de cada entrega para cada centro.
  4.   Resolve um Problema de Otimização (MIP):
  5.     OBJETIVO: Minimizar a distância total das entregas aos seus centros.
  6.     RESTRIÇÃO 1: Cada entrega deve ser atribuída a exatamente um cluster.
  7.     RESTRIÇÃO 2: A soma dos tamanhos das entregas em cada cluster não pode exceder a capacidade do veículo.
  8.   Obtém as novas atribuições de entrega para cada cluster a partir da solução do MIP.
  9.   Recalcula os centros de cada cluster (média ponderada das localizações das entregas atribuídas).
  10.  SE a mudança na posição dos centros for muito pequena, INTERROMPE o loop.
  11. RETORNA as atribuições finais.
```

#### `GREEDY` (Agrupamento Guloso)
Uma heurística simples e rápida. Primeiro, ordena as entregas da mais distante para a mais próxima do depósito. Depois, percorre essa lista e atribui cada entrega ao primeiro veículo que encontrar com capacidade livre.

```plaintext
FUNÇÃO FirstFit(entregas, veículos):
  1. Calcula a distância de cada entrega ao depósito.
  2. Ordena a lista de entregas em ordem decrescente de distância.
  3. Para cada veículo, inicializa sua capacidade restante.
  4. PARA cada entrega na lista ordenada:
  5.   PARA cada veículo na lista de veículos:
  6.     SE o veículo tem capacidade restante >= tamanho da entrega:
  7.       Atribui a entrega ao veículo.
  8.       Subtrai o tamanho da entrega da capacidade restante do veículo.
  9.       PASSA para a próxima entrega.
  10. RETORNA as atribuições.
```

### 2. Algoritmos de Roteamento (Routing)

*O objetivo é encontrar a melhor sequência de visitas para um grupo de entregas já atribuído a um veículo.*

#### `BRKGA` (Biased Random-Key Genetic Algorithm)
Uma meta-heurística poderosa baseada em evolução que busca uma solução de alta qualidade para o Problema do Caixeiro Viajante com Janelas de Tempo.

```plaintext
FUNÇÃO BRKGA_Routing(grupo_de_entregas):
  1. Inicializa uma "população" de N soluções. Cada solução é um vetor de chaves aleatórias, com uma chave para cada entrega.
  2. PARA cada geração de 1 a max_gens:
  3.   PARA cada solução (vetor de chaves) na população:
  4.     Decodifica a solução em uma sequência de visitas (ordenando as entregas pelas suas chaves).
  5.     Calcula a "aptidão" da sequência (custo total, combinando tempo de rota e penalidades por atraso).
  6.   Ordena a população, das soluções com melhor aptidão para as piores.
  7.   SE a melhor solução desta geração for melhor que a melhor já encontrada, salva-a e reseta o contador de "não melhoria".
  8.   SENÃO, incrementa o contador. SE o contador atingir um limite, INTERROMPE.
  9.   Cria a próxima geração:
  10.    Mantém um percentual das melhores soluções (a "elite").
  11.    Gera novas soluções cruzando uma solução da elite com uma não-elite. Há uma alta probabilidade de herdar a chave da elite.
  12.    Adiciona algumas soluções totalmente novas e aleatórias (mutantes).
  13. Pega a melhor sequência já encontrada.
  14. Aplica heurísticas de Busca Local (2-opt, or-opt, relocate) para refinar ainda mais a sequência.
  15. RETORNA a sequência final otimizada.
```

#### `GREEDY` (Inserção Mais Barata)
Uma heurística clássica que constrói a rota passo a passo.

```plaintext
FUNÇÃO CheapestInsertion(grupo_de_entregas):
  1. Inicia a rota com a entrega mais próxima do depósito.
  2. Cria uma lista de entregas não visitadas.
  3. ENQUANTO houver entregas não visitadas:
  4.   Inicializa custo_minimo = infinito.
  5.   Inicializa melhor_entrega_para_inserir e melhor_posicao.
  6.   PARA cada entrega_nao_visitada:
  7.     PARA cada posicao na rota atual (incluindo início e fim):
  8.       Calcula o custo_de_insercao = tempo(ponto_anterior, entrega_nao_visitada) + tempo(entrega_nao_visitada, ponto_seguinte) - tempo(ponto_anterior, ponto_seguinte).
  9.       SE custo_de_insercao < custo_minimo:
  10.        Atualiza custo_minimo, melhor_entrega_para_inserir e melhor_posicao.
  11.  Insere a melhor_entrega_para_inserir na melhor_posicao da rota.
  12.  Remove a entrega inserida da lista de não visitadas.
  13. RETORNA a rota final.
```

### 3. Algoritmos Únicos / Híbridos

*Resolvem o problema de atribuição e roteamento em uma única etapa.*

#### `MANUAL` (Atribuição Manual)
Uma heurística muito simples, baseada em regras para agrupar entregas. A ordem da rota não é otimizada.

```plaintext
FUNÇÃO ManualAssignment(entregas, veículos):
  1. Para cada entrega, calcula a "folga" = prazo_final - tempo_de_viagem_do_deposito.
  2. Ordena as entregas pela menor folga (mais urgentes primeiro).
  3. Ordena os veículos pela maior capacidade.
  4. PARA cada veículo na lista ordenada:
  5.   ENQUANTO o veículo tiver capacidade sobrando e houver entregas disponíveis:
  6.     Seleciona a entrega mais urgente ainda não atribuída (semente) e adiciona ao veículo.
  7.     Percorre as demais entregas não atribuídas (candidatas):
  8.       SE a candidata couber no veículo E estiver dentro de um raio de tempo limite do depósito (ex: 8 min):
  9.         Adiciona a candidata ao veículo.
  10.    SE o veículo encher, passa para o próximo veículo.
  11. RETORNA as rotas geradas.
```

#### `GREEDY_INSERTION` (Inserção Gulosa Híbrida)
Uma estratégia gulosa que, a cada passo, faz a melhor inserção possível em todo o sistema.

```plaintext
FUNÇÃO GreedyInsertionHybrid(entregas, veículos):
  1. Inicializa rotas vazias para todos os veículos.
  2. ENQUANTO houver entregas não alocadas:
  3.   Inicializa aumento_de_custo_minimo = infinito.
  4.   Inicializa melhor_insercao = nulo.
  5.   PARA cada entrega_nao_alocada:
  6.     PARA cada veiculo:
  7.       SE o veículo tem capacidade para a entrega:
  8.         custo_atual = Avalia(rota_atual_do_veiculo).
  9.         PARA cada posicao na rota do veículo:
  10.          nova_rota = Insere a entrega na posição.
  11.          custo_novo = Avalia(nova_rota).
  12.          aumento_de_custo = custo_novo - custo_atual.
  13.          SE aumento_de_custo < aumento_de_custo_minimo:
  14.            Atualiza aumento_de_custo_minimo e guarda a melhor_insercao (veículo, posição, entrega).
  15.  SE uma melhor_insercao foi encontrada:
  16.    Aplica a melhor_insercao (adiciona a entrega à rota correspondente).
  17.    Remove a entrega da lista de não alocadas.
  18.  SENÃO (nenhuma inserção possível), INTERROMPE.
  19. RETORNA as rotas finais.
```

#### `BRKGA_UNIQUE` (BRKGA para VRP)
O algoritmo mais sofisticado do sistema, que usa um algoritmo genético para resolver o problema completo de forma integrada.

```plaintext
FUNÇÃO BRKGA_Unique(entregas, veículos):
  1. Inicializa uma "população" de N cromossomos. Cada cromossomo é um vetor de chaves aleatórias, uma para cada entrega.
  2. PARA cada geração de 1 a max_gens:
  3.   PARA cada cromossomo na população:
  4.     // Decodifica o cromossomo em uma solução completa
  5.     Ordena as entregas de acordo com suas chaves no cromossomo (lista de prioridade).
  6.     Inicializa rotas vazias para todos os veículos.
  7.     PARA cada entrega na lista de prioridade:
  8.       // Usa uma heurística de inserção gulosa para alocá-la
  9.       Encontra a melhor posição, em qualquer rota de qualquer veículo, que minimize o aumento de custo.
  10.      Insere a entrega nesse local.
  11.    Calcula a "aptidão" da solução final (custo total de todas as rotas + penalidades por entregas não alocadas).
  12.  // Evolui a população (mesma lógica do BRKGA_Routing)
  13.  Ordena a população pela aptidão.
  14.  Gera a próxima geração com elites, cruzamento e mutantes.
  15. RETORNA a melhor solução completa encontrada.
```
