# Segmentação de Espinhas em Faces com U-Net

Este projeto tem como objetivo realizar a **segmentação automática de espinhas** em imagens faciais, utilizando redes neurais convolucionais (CNNs) e técnicas de visão computacional.



## ✏️  Descrição do Problema

A identificação de espinhas em imagens de rosto humano é desafiadora devido ao tamanho da região de cada imagem, variação de iluminação, tons de pele e presença de ruídos. Este trabalho visa detectar **regiões com espinhas** a partir de imagens de resolução convencional (smartphones), utilizando **máscaras binárias** geradas manualmente como referência para o treinamento.

O modelo utilizado foi uma U-Net personalizada, com ajustes para capturar **estruturas pequenas e sutis da pele**.

---

## ⚙️ Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
pip install -r requirements.txt
```

Certifique-se de ter Python 3.8+ e uma GPU com CUDA (opcional, mas recomendado).

---

## 🚀 Treinamento

1. **Pré-processamento:** As imagens brutas foram organizadas e renomeadas via `preprocessing.py`.

2. **Anotação:** Um subconjunto das imagens de treino foi anotado com a ferramenta [VGG VIA](https://www.robots.ox.ac.uk/~vgg/software/via/), e as máscaras foram geradas com `train_masks.py` e `val_masks.py`.

3. **Divisão dos dados:** O conjunto foi dividido em treino (70%), validação (15%) e teste (15%) com controle de semente. Apenas imagens originais estão presentes no conjunto de teste.

4. **Treinamento:** Execute o seguinte comando para treinar o modelo:

```bash
python train.py
```

Durante o treinamento:
- As predições sobre o conjunto de validação são salvas em `val_preds_vis/`.
- As curvas de desempenho (`loss_curve.png` e `dice_curve.png`) são geradas.

---

## 📊 Principais Resultados

Após 20 épocas de treinamento com `UNetCustom`:

| Métrica    | Treino   | Validação |
|------------|----------|-----------|
| Dice       | ~0.41    | ~0.39     |
| IoU        | ~0.28    | ~0.25     |
| Precision  | ~0.54    | ~0.48     |
| Recall     | ~0.34    | ~0.32     |

> Até o momento, as máscaras geradas destacam regiões da face, sendo necessário aprimorar o projeto para as regiões de espinhas. As métricas obtidas indicam que o modelo é capaz de identificar algumas regiões com espinhas, mas ainda apresenta desempenho limitado.
> O Dice (~0.39) e o IoU (~0.25) refletem a dificuldade de segmentar corretamente objetos pequenos e pouco representados. O modelo demonstrou boa precisão (~0.48), evitando falsos positivos, porém com baixo recall (~0.32), o que sugere que muitas espinhas reais não foram detectadas.
> Esses resultados reforçam a necessidade de ampliar a base anotada, refinar o modelo e explorar estratégias específicas para objetos pequenos.

---

## 🧠 Estratégia de Anotação e Divisão

- **Anotação:** Como o dataset original não possuía labels, foi feita anotação manual de um subconjunto de imagens usando a ferramenta VIA (VGG Image Annotator). As máscaras geradas são binárias, com **255 nas regiões com espinhas**.

- **Divisão:** A separação entre treino, validação e teste foi baseada em uma distribuição 70/15/15 sobre **todas as imagens disponíveis**, garantindo que **nenhuma imagem de teste contenha dados aumentados**.

> Imagens anotadas manualmente foram usadas apenas em `train_sub` e `val_sub`, para garantir qualidade das métricas.

---
## ✨ Futuro

- Aplicar pós-processamento para reduzir falsos positivos  
- Avaliar com métricas adicionais como mAP  
- Anotar mais imagens para refinar o treinamento  
- Aprimorar a rede U-Net para suavizar as perdas das pequenas regiões, com maior atenção para as ocorrências de acne
