Attention Mechanism - Mecanismo de Atenção: Uma inovação arquitetônica que permite à rede neural acessar e ponderar seletivamente diferentes partes de toda a sequência de entrada em cada etapa de geração. Surgiu para resolver as limitações das antigas Redes Neurais Recorrentes (RNNs), que dependiam de um único "estado oculto" e perdiam o contexto em textos longos.

Self-Attention - Autoatenção: Um mecanismo onde o "auto" se refere à capacidade do modelo de calcular pesos de atenção relacionando diferentes posições dentro de uma única sequência de entrada (como as palavras em uma mesma frase). Ele avalia as dependências entre várias partes da própria entrada para criar representações enriquecidas, diferentemente da atenção tradicional focada em sequências distintas (como entrada versus saída).

Attention Scores - Pontuações de Atenção: Valores intermediários calculados através do produto escalar entre os elementos da sequência de entrada. Como o produto escalar é uma medida de similaridade, uma pontuação mais alta indica um grau maior de alinhamento ou importância entre duas palavras específicas.

Attention Weights - Pesos de Atenção: São as pontuações de atenção após passarem pela etapa de normalização usando a função softmax. Esta normalização garante que os pesos sejam sempre positivos e que a soma deles resulte em 1 (ou 100%), permitindo que sejam interpretados como probabilidades ou graus de importância relativa.

Context Vector - Vetor de Contexto: O resultado final do mecanismo de autoatenção para um dado elemento. É um embedding (representação vetorial) enriquecido, calculado como uma soma ponderada de todos os vetores de entrada multiplicados pelos seus respectivos pesos de atenção.

Query / Key / Value - Consulta / Chave / Valor: Conceitos emprestados da área de banco de dados e recuperação de informações, usados na auto atenção com pesos treináveis.
Query - Consulta: Representa o token atual no qual o modelo está focado, usado para sondar outras partes da sequência de entrada.
Key - Chave: Cada token de entrada possui uma chave associada, que é usada para corresponder e calcular a similaridade com a Consulta.
Value - Valor: Representa o conteúdo real da entrada; uma vez que o modelo determina a relevância das chaves, ele recupera os Valores correspondentes para compor o vetor de contexto.

Scaled Dot-Product Attention - Atenção por Produto Escalar Escalonado: Uma técnica em que as pontuações de atenção são divididas (escalonadas) pela raiz quadrada da dimensão de incorporação (embedding dimension). Isso é feito porque, em dimensões muito altas (comuns em LLMs), grandes produtos escalares fariam a função softmax agir como uma "função degrau", resultando em gradientes próximos de zero e estagnando o aprendizado durante o treinamento.

Causal Attention - Atenção Causal / Máscara Causal: Uma forma especializada de autoatenção essencial para a geração de texto. Ela restringe o modelo a considerar apenas os tokens anteriores e o token atual, mascarando os tokens futuros. Na prática matemática, os pesos de atenção acima da diagonal principal da matriz são substituídos por infinito negativo (-ထ), para que a função softmax os transforme em probabilidade zero, garantindo que não haja "vazamento" de informações futuras.

Dropout: Uma técnica de aprendizado profundo onde algumas unidades (como uma porcentagem dos pesos de atenção) são ignoradas ou zeradas aleatoriamente durante o treinamento. Isso ajuda a evitar o sobreajuste (overfitting), garantindo que a rede não dependa excessivamente de conexões específicas para fazer suas previsões.

Multi-Head Attention - Atenção Multi-Cabeça: A prática de dividir o mecanismo de atenção em várias "cabeças" operando de forma paralela e independente. Cada cabeça aprende diferentes aspectos dos dados, permitindo que o modelo preste atenção simultaneamente a informações de diferentes subespaços de representação em posições diferentes, melhorando a performance em tarefas complexas.
