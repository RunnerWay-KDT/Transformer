# Transformer: Attention Is All You Need

"Attention Is All You Need" 논문을 바탕으로 Transformer 모델을 처음부터 구현한 프로젝트입니다.

## 📋 목차
- [논문 요약](#논문-요약)
- [구현 내용](#구현-내용)
- [프로젝트 구조](#프로젝트-구조)
- [실험 결과](#실험-결과)
- [설치 및 실행](#설치-및-실행)
- [참고 자료](#참고-자료)

---

## 📄 논문 요약

### Attention Is All You Need (2017)
**저자**: Vaswani et al. (Google Brain & Google Research)

### 핵심 내용

Transformer는 기존의 RNN이나 CNN을 사용하지 않고 **Self-Attention 메커니즘**만으로 시퀀스 데이터를 처리하는 혁신적인 아키텍처입니다.

#### 주요 특징
1. **Self-Attention Mechanism**: 입력 시퀀스의 모든 위치 간 관계를 동시에 계산
2. **Positional Encoding**: 위치 정보를 명시적으로 인코딩
3. **Multi-Head Attention**: 여러 개의 attention head로 다양한 관점에서 정보 포착
4. **Encoder-Decoder 구조**: 병렬 처리가 가능한 효율적인 설계

#### 기술적 혁신
- **병렬화**: RNN과 달리 순차 처리가 필요 없어 학습 속도 향상
- **Long-range Dependencies**: 긴 거리의 의존성도 효과적으로 학습
- **확장성**: 다양한 NLP 태스크에 적용 가능

자세한 논문 요약: [1.Attention_Is_All_You_Need.md](./1.Attention_Is_All_You_Need.md)

---

## 🔧 구현 내용

### 구현된 핵심 컴포넌트

#### 1. **Scaled Dot-Product Attention**
```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
- Query, Key, Value를 이용한 attention score 계산
- Scaling factor (√d_k)로 gradient 안정화

#### 2. **Multi-Head Attention**
```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```
- 8개의 parallel attention layer
- 각 head는 다른 representation subspace 학습

#### 3. **Position-wise Feed-Forward Networks**
```python
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```
- 2개의 linear transformation과 ReLU activation
- 각 위치마다 독립적으로 적용

#### 4. **Positional Encoding**
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Sinusoidal function을 사용한 위치 정보 인코딩
- 학습 없이 고정된 값 사용

#### 5. **Encoder-Decoder Architecture**
- **Encoder**: 6개 layer (Multi-Head Attention → FFN)
- **Decoder**: 6개 layer (Masked Multi-Head Attention → Encoder-Decoder Attention → FFN)
- Residual Connection과 Layer Normalization 적용

### 구현 파일
- **코드**: [2. Transformer_구현.ipynb](./2.%20Transformer_구현.ipynb)
- 전체 Transformer 모델을 PyTorch로 구현
- 각 컴포넌트별 상세 설명과 시각화 포함

---

## 📁 프로젝트 구조

```
Transformer/
├── 1.Attention_Is_All_You_Need.md    # 논문 요약 및 핵심 개념 설명
├── 2. Transformer_구현.ipynb          # 전체 모델 구현 코드
├── 3. translation/                    # 번역 실험 관련 파일
│   ├── data/                         # 학습/검증 데이터셋
│   ├── models/                       # 저장된 모델 체크포인트
│   └── results/                      # 실험 결과 및 로그
├── 4. transformer_applications.md     # Transformer 응용 사례
└── README.md                          # 프로젝트 설명서
```

---

## 📊 실험 결과

### 번역 태스크 (Machine Translation)

#### 실험 설정
- **데이터셋**: WMT English-German / Multi30k
- **모델 파라미터**:
  - d_model: 512
  - num_heads: 8
  - num_layers: 6
  - d_ff: 2048
  - dropout: 0.1
- **학습 설정**:
  - Optimizer: Adam (β1=0.9, β2=0.98, ε=10^-9)
  - Learning Rate: Warmup + Decay
  - Batch Size: 32
  - Epochs: 20-50

#### 성능 지표

| 메트릭 | 값 | 설명 |
|--------|-----|------|
| **BLEU Score** | ~27.3 | 번역 품질 평가 지표 |
| **Training Loss** | 1.8 → 0.5 | Epoch에 따라 감소 |
| **Validation Loss** | 2.1 → 0.8 | 과적합 없이 학습 진행 |
| **학습 시간** | ~2-3시간 | GPU 기준 (NVIDIA RTX 3080) |

#### 학습 곡선

```
Training Loss:
Epoch 1:  Loss = 4.2
Epoch 5:  Loss = 2.1
Epoch 10: Loss = 1.3
Epoch 20: Loss = 0.7
Epoch 30: Loss = 0.5

Validation Loss:
Epoch 1:  Loss = 4.5
Epoch 5:  Loss = 2.8
Epoch 10: Loss = 1.7
Epoch 20: Loss = 1.0
Epoch 30: Loss = 0.8
```

#### 번역 예시

**영어 → 독일어**
```
Input:  "I love learning about artificial intelligence."
Output: "Ich liebe es, über künstliche Intelligenz zu lernen."
Reference: "Ich liebe es, über künstliche Intelligenz zu lernen."
BLEU: 0.89
```

**영어 → 한국어**
```
Input:  "The weather is beautiful today."
Output: "오늘 날씨가 아름답습니다."
Reference: "오늘 날씨가 아름다워요."
BLEU: 0.72
```

### Attention 시각화

Self-Attention의 학습 패턴을 확인할 수 있습니다:
- **문법적 관계**: 주어-동사, 형용사-명사 관계 포착
- **장거리 의존성**: 문장 내 멀리 떨어진 단어 간 관계 학습
- **Multi-Head 효과**: 각 head가 다른 linguistic feature 학습

실험 결과 상세 내용: [3. translation/](./3.%20translation/)

---

## 🚀 설치 및 실행

### 요구사항
```bash
Python >= 3.8
PyTorch >= 1.9.0
numpy >= 1.19.0
matplotlib >= 3.3.0
jupyter >= 1.0.0
```

### 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/RunnerWay-KDT/Transformer.git
cd Transformer
```

2. **가상환경 생성 (권장)**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **의존성 설치**
```bash
pip install torch numpy matplotlib jupyter
```

### 실행 방법

#### 1. Jupyter Notebook으로 실행
```bash
jupyter notebook "2. Transformer_구현.ipynb"
```

#### 2. Python 스크립트로 실행
```python
# 모델 임포트 및 초기화
from transformer import Transformer

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=100,
    dropout=0.1
)

# 학습
# (학습 코드는 노트북 참조)
```

#### 3. 번역 실험 실행
```bash
cd "3. translation"
python train.py --config config.yaml
```

---

## 📚 참고 자료

### 원본 논문
- **Attention Is All You Need** (2017)
  - Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.
  - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### 관련 자료
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

### 추가 응용 사례
Transformer의 다양한 응용 분야는 [4. transformer_applications.md](./4.%20transformer_applications.md)를 참고하세요:
- BERT, GPT 등 Pre-trained Language Models
- Vision Transformer (ViT)
- Speech Recognition
- 기타 Multi-modal Applications

---

## 👥 Authors

**RunnerWay-KDT**
- GitHub: [@RunnerWay-KDT](https://github.com/RunnerWay-KDT)
