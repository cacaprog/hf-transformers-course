## Capítulo 1
- Os pipelines consomem VRAM, por isso, após cada task é importante deletar a variável e coletar o lixo
- Os Transformers podem combinar dados de múltiplas fontes (áudio, texto, imagem)
- Algumas tasks não funcionam mais dentro do pipeline, para realizá-las, precisei carregar o modelo e o tokenizer manualmente

> continuar em  How do Transformers work? 
> https://huggingface.co/learn/llm-course/en/chapter1/4

## Captítulo 2
- foram apresentados 3 frameworks para o deploy dos modelos: TGI, llma.cpp e vLLM
- o ideal é que o modelo e o tokenizer sejam do mesmo checkpoint
- o "head" do modelo é responsável por traduzir o modelo para as tarefas específicas
- é importante entender sobre a tarefa que será realizada, qual o peso que isso tem, qual o volume, para dimencionar adequadamente o modelo e arquitetura com o hardware
- a escolha do  framework ajuda nesta tarefa, inclusive para controlar os parâmetros adequadamente
- os modelos GGUF podem ser rodados em devices, docker, além da máquina. Boa alternativa para ttarefas específicas que podem utilizar modelos menores

